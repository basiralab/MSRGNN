import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvModule(nn.Module):
    def __init__(
            self, 
            in_channels=1, 
            out_channels=32, 
            kernel_size=3, 
            stride=2, 
            padding=0):
        super(ConvModule, self).__init__()

        self.conv_blocks = nn.ModuleList()
        self.out_channels = out_channels

        self.conv_blocks.append(
            ConvBnRelu(in_channels, out_channels, kernel_size, stride, padding)
        )
        for _ in range(3):
            self.conv_blocks.append(
                ConvBnRelu(out_channels, out_channels, kernel_size, stride, padding)
            )
    
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # return x.view(-1, 16, self.out_channels*4*4)
        return x.view(x.size(0), -1)
    
class RelationModule(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=512, out_channels=256):
        super(RelationModule, self).__init__()
        self.fc1 = nn.Linear(in_channels*2, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, out_channels)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=13, dropout=0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # return x.view(-1, 8, self.output_dim)
        return x
    
class PanelEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super(PanelEmbedding, self).__init__()
        self.in_dim = in_dim
        self.fc = nn.Linear(self.in_dim, out_dim)

    def forward(self, x):
        return self.fc(x.view(-1, self.in_dim))

class WReN(nn.Module):
    def __init__(
            self,
            use_tag = True,
            image_size = 80
    ):
        super(WReN, self).__init__()

        self.image_size = image_size
        self.use_tag = use_tag

        self.conv = ConvModule()
        self.relation = RelationModule()
        # self.mlp = MLP()
        # self.proj = PanelEmbedding(tag=use_tag)
        # self.tags = self.tag_panels()


    def tag_panels(self):
        tags = []
        for i in range(16):
            tag = np.zeros([1, 9], dtype=float)
            tag[:, i if i < 8 else 8] = 1.0
            tag_tensor = torch.tensor(tag, dtype=torch.float32).unsqueeze(0)
            tags.append(tag_tensor.cuda() if torch.cuda.is_available() else tag_tensor)
        return torch.cat(tags, dim=1)


    def group_panel_embeddings(self, embeddings):
        embeddings = embeddings.view(-1, 16, 256)
        context = embeddings[:, :8, :]
        candidates = embeddings[:, 8:, :]

        context_i = context.unsqueeze(1).expand(-1, 8, -1, -1)
        context_j = context.unsqueeze(2).expand(-1, -1, 8, -1)

        context_pairs = torch.cat(
            [context_i, context_j], dim=3
        ).view(-1, 64, 512)

        context_embeddings = context_i
        candidate_embeddings = candidates.unsqueeze(2).expand(-1, -1, 8, -1)

        context_candidate_pair_i = torch.cat([context_embeddings, candidate_embeddings], dim=3)
        context_candidate_pair_j = torch.cat([candidate_embeddings, context_embeddings], dim=3)

        embeddings_pairs = [
            context_pairs.unsqueeze(1).expand(-1, 8, -1, -1),
            context_candidate_pair_i,
            context_candidate_pair_j
        ]

        return torch.cat(embeddings_pairs, dim=2).view(-1, 8, 80, 512)


    def relation_network_sum_features(self, features):
        features = features.view(-1, 8, 80, 256)
        return torch.sum(features, dim=2)
    
    def compute_loss(self, pred, target):
        return F.cross_entropy(pred, target)
    
    def forward(self, x):
        panel_features = self.conv(x.view(-1, 1, self.image_size, self.image_size))

        if self.use_tag:
            tags_for_batch = self.tags.expand(x.size(0), -1, -1)
            panel_features = torch.cat([panel_features, tags_for_batch], dim=2)

        panel_embeddings = self.proj(panel_features)
        pairwise = self.group_panel_embeddings(panel_embeddings)
        
        relation_features = self.relation(pairwise.view(-1, 512))
        relation_features = self.relation_network_sum_features(relation_features)

        output = self.mlp(relation_features.view(-1, 256))

        prediction = output[:, :, 12]

        # return output.squeeze(-1)
        return prediction
    
class OddOneOutWReN(WReN):
    def __init__(self, num_total_panels, image_size=80):
        super().__init__(image_size=image_size)
        self.tag_dim = num_total_panels
        conv_feature_dim = self.conv.out_channels * 4 * 4 
        embedding_dim = 256

        self.proj = PanelEmbedding(in_dim=conv_feature_dim + self.tag_dim, out_dim=256)

        self.mlp = MLP(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=1
        )

    def forward(self, x):
        B, P, C, H, W = x.shape
        D_emb = 256

        x = self.conv(x.view(B*P, C, H, W))
        tags = torch.eye(P, device=x.device).expand(B, -1, -1).reshape(B * P, P)
        combined_feats = torch.cat(
            [x, tags], dim=1
        )

        panel_emb = self.proj(combined_feats).view(B, P, D_emb)

        mask = ~torch.eye(P, dtype=torch.bool, device=x.device)
        x_scenarios = torch.stack(
            [
                panel_emb[b, mask[i]] for b in range(B) for i in range(P)
            ]
        ).view(B, P, P - 1, D_emb)

        batched_scenarios = x_scenarios.view(B* P, P-1, D_emb)

        panel_i = batched_scenarios.unsqueeze(2).expand(-1, -1, P-1, -1)
        panel_j = batched_scenarios.unsqueeze(1).expand(-1, P-1, -1, -1)

        pairs = torch.cat(
            [panel_i, panel_j], dim=3
        )

        relation_output = self.relation(pairs.view(-1, 2 * D_emb))

        relation_output_grouped = relation_output.view(B * P, (P-1)**2, D_emb)

        aggregated_feats = torch.sum(
            relation_output_grouped,
            dim=1
        )

        scores = self.mlp(aggregated_feats)

        return scores.view(B, P)

def train_epoch(model, loader, optimizer, criterion, device, scaler=None, scheduler=None):
    model.train(); total_loss=0; total_correct = 0; total_samples = 0
    for imgs, labels in tqdm(loader, desc="Training"): # Added tqdm
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Use autocast for mixed precision
                scores = model(imgs)
                loss = criterion(scores, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            scores = model(imgs)
            loss = criterion(scores, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * imgs.size(0)

        preds = scores.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    # print(f"Train Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

    # return total_loss / len(loader.dataset)
    return average_loss, accuracy

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval(); total_loss=0; correct=0; total=0
    for imgs, labels in tqdm(loader, desc="Validating"): # Added tqdm
        imgs, labels = imgs.to(device), labels.to(device)
        scores = model(imgs)
        loss = criterion(scores, labels)
        preds = scores.argmax(dim=1)
        correct += (preds==labels).sum().item(); total += labels.size(0)
        total_loss += loss.item() * imgs.size(0)
    return total_loss/len(loader.dataset), correct/total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)