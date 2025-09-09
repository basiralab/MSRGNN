from tensorboardX import SummaryWriter
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
from torch.nn import LayerNorm
from torch.utils.data import DataLoader
from tqdm import tqdm 
import argparse
import itertools
import json
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# --- 1. Graph Structure Helpers ---

def create_row_col_template(num_nodes=9):
    """
    Creates the edge_index template for a single 3x3 grid, connecting
    nodes along rows and columns.
    """
    if num_nodes != 9:
        raise ValueError("This template is specifically for a 3x3 grid (9 nodes).")

    rows = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    cols = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]

    single_graph_edges = set()
    for group in rows + cols:
        permutations = list(itertools.permutations(group, 2))
        for p in permutations:
            single_graph_edges.add(p)

    if not single_graph_edges:
        return torch.empty((2, 0), dtype=torch.long)
        
    return torch.tensor(list(single_graph_edges), dtype=torch.long).t()

def create_fully_connected_template(num_nodes=9):
    """
    Creates the edge_index template for a single fully-connected graph.
    """
    nodes = torch.arange(num_nodes)
    permutations = list(itertools.permutations(nodes.tolist(), 2))
    
    if not permutations:
        return torch.empty((2, 0), dtype=torch.long)
        
    return torch.tensor(permutations, dtype=torch.long).t()

def batch_template_edge_index(template_edge_index, num_graphs, num_nodes_per_graph, device):
    """
    Takes a template edge_index for a single graph and batches it for use in PyG.
    """
    template_edge_index = template_edge_index.to(device)
    offsets = torch.arange(0, num_graphs, device=device) * num_nodes_per_graph
    edge_offsets = offsets.repeat_interleave(template_edge_index.size(1))
    return template_edge_index.repeat(1, num_graphs) + edge_offsets


# --- 2. Core Model Components ---

class RelationNetPairwiseV2(nn.Module):
    """MLP that processes a pair of node features to produce a relation vector."""
    def __init__(self, input_dim, hidden_dim, num_mlp_layers=2, dropout_rate=0.1):
        super().__init__()
        layers = []
        current_dim = 2 * input_dim
        for _ in range(num_mlp_layers - 1):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_i, x_j):
        return self.mlp(torch.cat([x_j, x_i], dim=-1))

class BasicBlock(nn.Module):
    """A basic residual block for the ResNet backbone."""
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, stride, bias=False), 
            nn.BatchNorm2d(out_planes)
        ) if stride != 1 or in_planes != out_planes else None

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.gelu(out + identity)

class ResNetFeatureExtractor(nn.Module):
    """
    A ResNet-based feature extractor that produces a dictionary of multi-scale
    feature vectors for each input image.
    """
    def __init__(self, in_ch=1, base_channels=64):
        super().__init__()
        self.in_planes = base_channels
        self.conv1 = nn.Conv2d(in_ch, base_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_channels, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks=2, stride=2)
        
        self.pools = nn.ModuleDict({
            'pool_4x4': nn.AdaptiveAvgPool2d(4),
            'pool_2x2': nn.AdaptiveAvgPool2d(2),
            'pool_1x1': nn.AdaptiveAvgPool2d(1)
        })
        
        final_conv_channels = base_channels * 4
        self.normalizers = nn.ModuleDict()
        self.fdim_dict = {}
        for name, pool in self.pools.items():
            pool_size = pool.output_size
            dim = final_conv_channels * pool_size * pool_size
            self.normalizers[name] = nn.LayerNorm(dim)
            self.fdim_dict[name] = dim
            
    def _make_layer(self, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, out_planes, s))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        features = {}
        for name, pool in self.pools.items():
            pooled = pool(out).view(x.size(0), -1)
            normalized = self.normalizers[name](pooled)
            features[name] = normalized
        return features

class GatedGNNRelationLayerV2(MessagePassing):
    """
    A Gated Graph Attention layer. The message content is generated by a
    RelationNet, and the attention weights are computed separately.
    It explicitly uses visual features and positional embeddings.
    """
    def __init__(self, feature_dim, pos_dim, out_channels,
                 relation_net_mlp_layers, relation_net_dropout, gnn_message_dropout_p):
        super().__init__(aggr='add', node_dim=0)
        integrated_dim = feature_dim + pos_dim
        
        self.pairwise_fn = RelationNetPairwiseV2(
            integrated_dim, out_channels, relation_net_mlp_layers, relation_net_dropout
        )
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim + 2 * pos_dim, out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, 1)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(integrated_dim + out_channels, out_channels),
            nn.GELU()
        )
        self.norm = LayerNorm(out_channels)
        self.skip_proj = nn.Linear(integrated_dim, out_channels) if integrated_dim != out_channels else nn.Identity()
        self.gnn_message_dropout_p = gnn_message_dropout_p

    def forward(self, x, pos_emb, edge_index):
        integrated_x = torch.cat([x, pos_emb], dim=-1)
        aggr_out = self.propagate(edge_index, x=x, pos_emb=pos_emb, integrated_x=integrated_x)
        updated_x = self.update_mlp(torch.cat([integrated_x, aggr_out], dim=-1))
        return self.norm(self.skip_proj(integrated_x) + updated_x)

    def message(self, x_i, x_j, pos_emb_i, pos_emb_j, integrated_x_i, integrated_x_j, edge_index_i):
        relational_message = self.pairwise_fn(integrated_x_j, integrated_x_i)
        
        attention_input = torch.cat([x_i, x_j, pos_emb_i, pos_emb_j], dim=-1)
        attention_logits = self.attention_mlp(attention_input)
        attention_weights = softmax(attention_logits, index=edge_index_i)
        
        gated_message = attention_weights * relational_message
        return F.dropout(gated_message, p=self.gnn_message_dropout_p, training=self.training)


# --- 3. High-Level Reasoning Modules ---

class SharedGNNReasoner(nn.Module):
    """
    A two-stage GNN reasoner.
    - Stage 1: A shared GNN layer processes each feature scale independently.
    - Stage 2: A unified GNN layer reasons over the fused outputs of Stage 1.
    Note: The GNN reasoner itself is agnostic to the graph structure, which is
    provided via the `edge_index` argument in the forward pass.
    """
    def __init__(self, fdim_dict, pos_emb_dim, gnn_hidden_dim, 
                 relation_net_mlp_layers, relation_net_dropout, 
                 gnn_message_dropout_p, reasoner_proj_dropout,
                 num_nodes_per_graph=9):
        super().__init__()
        self.panel_pos_emb = nn.Embedding(num_nodes_per_graph, pos_emb_dim)

        gnn_params = {
            'relation_net_mlp_layers': relation_net_mlp_layers,
            'relation_net_dropout': relation_net_dropout,
            'gnn_message_dropout_p': gnn_message_dropout_p
        }
            
        self.input_projections = nn.ModuleDict()
        common_gnn_dim = gnn_hidden_dim 
        for name, fdim in fdim_dict.items():
            self.input_projections[name] = nn.Linear(fdim, common_gnn_dim)
            
        self.shared_gnn_stage1 = GatedGNNRelationLayerV2(
            common_gnn_dim, pos_emb_dim, gnn_hidden_dim, **gnn_params
        )

        unified_input_dim = len(fdim_dict) * gnn_hidden_dim
        unified_output_dim = gnn_hidden_dim * 2 
        self.gnn_stage2 = GatedGNNRelationLayerV2(
            unified_input_dim, pos_emb_dim, unified_output_dim, **gnn_params
        )
        
        classifier_input_dim = unified_output_dim * 3 # for mean/max/add pooling
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, gnn_hidden_dim),
            nn.GELU(),
            nn.Dropout(reasoner_proj_dropout),
            nn.Linear(gnn_hidden_dim, 1)
        )

    def forward(self, node_features_dict, batch_indices, edge_index, num_nodes_per_graph=9):
        num_graphs = batch_indices.max().item() + 1
        device = batch_indices.device

        pos_ids = torch.arange(num_nodes_per_graph, device=device).repeat(num_graphs)
        pos_embs = self.panel_pos_emb(pos_ids)

        stage1_outputs = []
        for name, proj_layer in self.input_projections.items():
            features = node_features_dict[name]
            projected_features = proj_layer(features)
            processed_features = self.shared_gnn_stage1(projected_features, pos_embs, edge_index)
            stage1_outputs.append(processed_features)
            
        unified_h1 = torch.cat(stage1_outputs, dim=1)
        h2 = self.gnn_stage2(unified_h1, pos_embs, edge_index)
        
        pooled = torch.cat([
            global_mean_pool(h2, batch_indices),
            global_max_pool(h2, batch_indices),
            global_add_pool(h2, batch_indices)
        ], dim=-1)
        
        return self.classifier(pooled)


# --- 4. Top-Level Model ---

class MSRGNN(nn.Module):
    """
    Multi-Scale Relational Graph Neural Network.
    This model orchestrates feature extraction and graph-based reasoning.
    
    The graph structure is fixed at initialization. You can either specify a
    pre-defined `graph_type` ('row_col' or 'fully_connected') or provide your
    own `template_edge_index` for a single graph. The `template_edge_index`
    argument will always take precedence if provided.
    """
    def __init__(self, ctx_size=8, cand_size=8, resnet_base_channels=64,
                 pos_embedding_dim=128, gnn_hidden_dim=128, relation_net_mlp_layers=3,
                 relation_net_dropout=0.1, gnn_message_dropout_p=0.1,
                 reasoner_proj_dropout=0.3,
                 graph_type='row_col',
                 template_edge_index=None,
                 num_nodes_per_graph=9):
        super().__init__()
        
        # --- Determine and store the graph structure template ---
        final_template = None
        if template_edge_index is not None:
            if not isinstance(template_edge_index, torch.Tensor) or template_edge_index.dim() != 2 or template_edge_index.size(0) != 2:
                raise ValueError("`template_edge_index` must be a LongTensor of shape [2, num_edges].")
            final_template = template_edge_index
        elif graph_type is not None:
            if graph_type == 'row_col':
                final_template = create_row_col_template()
            elif graph_type == 'fully_connected':
                final_template = create_fully_connected_template()
            else:
                 raise ValueError(f"Unknown graph_type: '{graph_type}'.")
        else:
            raise ValueError("Must provide either `graph_type` or `template_edge_index`.")
        
        self.register_buffer('template_edge_index', final_template.long())

        self.extractor = ResNetFeatureExtractor(in_ch=1, base_channels=resnet_base_channels)
        self.ctx_size, self.cand_size = ctx_size, cand_size
        self.num_nodes_per_graph = num_nodes_per_graph

        self.reasoner = SharedGNNReasoner(
            fdim_dict=self.extractor.fdim_dict,
            pos_emb_dim=pos_embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            relation_net_mlp_layers=relation_net_mlp_layers,
            relation_net_dropout=relation_net_dropout,
            gnn_message_dropout_p=gnn_message_dropout_p,
            reasoner_proj_dropout=reasoner_proj_dropout,
            num_nodes_per_graph=num_nodes_per_graph
        )

    def forward(self, x, ctx_size, cand_size):
        if x.dim() == 4:
            B, P_total, H, W = x.shape; C_in = 1
        elif x.dim() == 5:
            B, P_total, C_in, H, W = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected 4D or 5D tensor.")

        # 1. Feature Extraction
        visual_features_dict = self.extractor(x.view(B * P_total, C_in, H, W))
        
        # 2. Data Structuring for GNN
        all_graphs_nodes_dict = {name: [] for name in self.extractor.fdim_dict.keys()}
        for name, features in visual_features_dict.items():
            fdim = self.extractor.fdim_dict[name]
            structured_features = features.view(B, P_total, fdim)
            ctx_features = structured_features[:, :self.ctx_size, :]
            cand_features = structured_features[:, self.ctx_size:, :]
            for i in range(self.cand_size):
                candidate_i = cand_features[:, i, :].unsqueeze(1)
                graph_nodes = torch.cat([ctx_features, candidate_i], dim=1)
                all_graphs_nodes_dict[name].append(graph_nodes)

        num_graphs = B * self.cand_size
        # num_nodes_per_graph = self.ctx_size + 1
        
        for name, graph_list in all_graphs_nodes_dict.items():
            stacked_graphs = torch.stack(graph_list).permute(1, 0, 2, 3)
            fdim = self.extractor.fdim_dict[name]
            flat_nodes = stacked_graphs.reshape(num_graphs, self.num_nodes_per_graph, fdim).view(-1, fdim)
            all_graphs_nodes_dict[name] = flat_nodes

        # 3. Graph Preparation
        batch_indices = torch.arange(
            num_graphs, device=x.device
        ).repeat_interleave(self.num_nodes_per_graph)
        
        # Batch the stored template for the current batch size
        edge_index = batch_template_edge_index(
            self.template_edge_index, num_graphs, self.num_nodes_per_graph, x.device
        )
        
        # 4. Relational Reasoning
        scores = self.reasoner(all_graphs_nodes_dict, batch_indices, edge_index)
        
        return scores.view(B, self.cand_size)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss=0; total_correct=0; total_samples = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        scores = model(x=imgs, ctx_size=8, cand_size=8)
        loss = criterion(scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (scores.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        
    average_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_samples
    print(f"Epoch Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    # return total_loss / len(loader.dataset), total_correct / total_samples
    return average_loss, accuracy

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval(); total_loss=0; correct=0; total=0
    for imgs, labels in tqdm(loader, desc="Validating"):
        imgs, labels = imgs.to(device), labels.to(device)
        scores = model(x=imgs, ctx_size=8, cand_size=8)
        loss = criterion(scores, labels)
        preds = scores.argmax(dim=1)
        correct += (preds==labels).sum().item(); total += labels.size(0)
        total_loss += loss.item() * imgs.size(0)
    return total_loss/len(loader.dataset), correct/total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
