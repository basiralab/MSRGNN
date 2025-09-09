from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import math
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# Based on https://github.com/mikomel/sal

class ConvBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualPreNormFeedForward(nn.Module):
    def __init__(
            self, 
            dim,
            expansion_factor = 4,
            dropout = 0.0,
            output_dim = None,
        ):
        super(ResidualPreNormFeedForward, self).__init__()
        output_dim = output_dim if output_dim else dim
        hidden_dim = dim * expansion_factor

        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(self.norm(x)) + x
        
class StructureAwareLayer(nn.Module):
    def __init__(
            self,
            out_channels,
            kernel_size=1,
            num_rows=6,
            num_cols=60
    ):
        super(StructureAwareLayer, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weights = nn.Parameter(
            torch.randn(num_rows, num_cols, out_channels, kernel_size)
        )

        self.biases = nn.Parameter(
            torch.randn(out_channels)
        )

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        if fan_in > 0:
            bound = 1/ math.sqrt(fan_in)
            nn.init.uniform_(self.biases, -bound, bound)


    # def forward(self, x, in_rows=3, in_cols=3):
    #     # print(f"Input shape: {x.shape}, in_rows: {in_rows}, in_cols: {in_cols}")
    #     batch_size, in_channels, seq_len = x.shape


    #     # print(self.weights.shape)
    #     num_rows, num_cols, _, _, = self.weights.shape

    #     row_step, col_step = num_rows // in_rows, num_cols // in_cols

    #     w = self.weights.unfold(0, row_step, row_step)
    #     w = w.unfold(1, col_step, col_step)

    #     w = w.mean(dim=(-1, -2))

    #     w = w.permute(2, 0, 1, 3)
    #     w= w.flatten(start_dim=1)

    #     kernel = w

    #     num_patches = seq_len // self.kernel_size

    #     x_patched = x.view(batch_size, in_channels, num_patches, self.kernel_size)
    #     x_patched = x_patched.transpose(1, 2)

    #     x_prepared = x_patched.flatten(start_dim=2)

    #     output = F.linear(x_prepared, kernel, self.biases)   

    #     return output.transpose(1, 2)  

    def forward(self, x, in_rows=3, in_cols=3):
        # print(f"Input shape: {x.shape}, in_rows: {in_rows}, in_cols: {in_cols}")
        w = self.weights.unfold(0, in_rows, in_rows)
        w = w.unfold(1, in_cols, in_cols)

        w = w.mean((0, 1))
        w = w.flatten(start_dim=1)
        w = w.transpose(0, 1)

        B, C_in, in_dim = x.shape
        num_groups = in_dim // self.kernel_size

        x = x.view(B, C_in, num_groups, self.kernel_size)

        x = x.transpose(1, 2).contiguous()
        x = x.flatten(0, 1)
        x = x.flatten(1, 2)

        x = torch.einsum(
            "bd,dc->bc",
            x,
            w
        ) + self.biases

        x = x.view(B, num_groups, self.out_channels)
        x = x.transpose(1, 2)

        return x


class SCAR(nn.Module):
    def __init__(
            self,
            hidden_dim=32,
            embedding_size=128,
            local_feature_dim=80,
            image_size=80,
            local_kernel_size=10,
            global_kernel_size=20,
            ff_dim=80,
            sal_rows = 6,
            sal_cols = 60
    ):
        super(SCAR, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.local_feature_dim = local_feature_dim
        self.image_size = image_size
        self.local_kernel_size = local_kernel_size
        self.global_kernel_size = global_kernel_size

        local_group_size  = ff_dim // local_kernel_size
        global_group_size = (local_kernel_size * 8) // global_kernel_size
        
        conv_dim = (40 * (image_size // 80)) ** 2

        self.local_model = nn.Sequential(
            ConvBlock(1, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(hidden_dim // 2, hidden_dim // 2 , kernel_size=3, padding=1),
            ConvBlock(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(conv_dim, ff_dim),
            nn.ReLU(inplace=True),
            ResidualPreNormFeedForward(ff_dim),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=128,
                kernel_size = (local_group_size,),
                stride=(local_group_size,)
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=128,
                out_channels=8,
                kernel_size=(1,),
                stride=(1,)
            ),
            nn.Flatten(start_dim=-2, end_dim=-1),
            ResidualPreNormFeedForward(local_kernel_size * 8)
        )

        self.sal = StructureAwareLayer(
            out_channels=64,
            kernel_size=global_group_size,
            num_rows=sal_rows,
            num_cols=sal_cols
        )

        self.global_model = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                kernel_size=(1,)
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=5,
                kernel_size=(1,)
            ),
            nn.Flatten(start_dim=-2, end_dim=-1),
            ResidualPreNormFeedForward(global_kernel_size * 5),
            nn.Linear(
                in_features=global_kernel_size * 5,
                out_features=embedding_size
            )
        )


    def forward(
        self,
        x,
        num_rows= -1,
        num_cols= 3,
        ctx_size=8,
        cand_size=8,
    ):
        B, P, C_in, H, W = x.shape
        num_rows = (ctx_size + 1) // 3

        x = x.view((B * P), C_in, H, W)
        x = self.local_model(x)
        x = x.view(B, P, -1)

        x = torch.cat(
            [
                x[:, :ctx_size, :]
                .unsqueeze(1)
                .repeat(1, cand_size, 1, 1),
                x[:, ctx_size:, :].unsqueeze(2),
            ],
            dim=2
        )

        x = x.view((B * cand_size), (ctx_size + 1), -1)

        x = self.sal(x, in_rows=num_rows, in_cols=num_cols)
        x = self.global_model(x)

        # print(f"Output shape: {x.shape}")

        x = x.view(B, cand_size, -1)

        # print(f"Final output shape: {x.shape}")
        return x
    
class RAVENSCAR(nn.Module):
    def __init__(self, **kwargs):
        super(RAVENSCAR, self).__init__()
        # print("Kwargs:", kwargs)
        self.scar = SCAR(**kwargs)

        # print(f"SCAR embedding size: {self.scar.embedding_size}")


        self.target_predictor = nn.Sequential(
            nn.Linear(self.scar.embedding_size, self.scar.embedding_size),
            nn.GELU(),
            nn.Linear(self.scar.embedding_size, 1),
            nn.Flatten(-2, -1)
        )

    def forward(self, x, num_rows=3, num_cols=3, ctx_size=8, cand_size=8):
        embeddings = self.scar(
            x, 
            num_rows=num_rows, 
            num_cols=num_cols, 
            ctx_size=ctx_size, 
            cand_size=cand_size
        )
        # print(f"Embeddings shape: {embeddings.shape}")
        targets = self.target_predictor(embeddings)
        # print(f"Output shape: {targets.shape}")
        return targets


class OddOneOutSCARBackbone(SCAR):
    def forward(
            self,
            x
    ):
        B, P, C_in, H, W = x.shape
        x = x.view((B * P), C_in, H, W)
        x = self.local_model(x)
        x = x.view(B, P, -1)

        embedding_dim = x.shape[-1]
        mask = (
            ~torch.eye(P, dtype=torch.bool, device=x.device)
            .unsqueeze(-1)
            .repeat(1, 1, embedding_dim)
        )

        x = torch.stack(
            [
                x.masked_select(m.repeat(B, 1, 1)).view(
                    B, P - 1, embedding_dim
                )
                for m in mask
            ],
            dim=1,
        )
        x = x.view((B * P), (P - 1), embedding_dim)
        x = self.sal(x, in_rows=1, in_cols=P-1)
        x = self.global_model(x)
        x = x.view(B, P, -1)
        return x
    

class OddOneOutSCAR(nn.Module):
    def __init__(self, **kwargs):
        super(OddOneOutSCAR, self).__init__()
        self.scar = OddOneOutSCARBackbone(**kwargs)

        self.target_predictor = nn.Sequential(
            nn.Linear(self.scar.embedding_size, self.scar.embedding_size),
            nn.GELU(),
            nn.Linear(self.scar.embedding_size, 1),
            nn.Flatten(-2, -1)
        )

    def forward(self, x):
        embeddings = self.scar(x)
        targets = self.target_predictor(embeddings)
        return targets

def train_epoch(model, loader, optimizer, criterion, device, scaler=None, scheduler=None):
    model.train(); total_loss=0; total_correct=0; total_samples = 0
    for imgs, labels in tqdm(loader, desc="Training"): # Added tqdm
        imgs, labels = imgs.to(device), labels.to(device)

        # print(labels.shape)  # Debugging line to check labels shape

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Use autocast for mixed precision
                # scores = model(x=imgs, num_rows=3, num_cols=3, ctx_size=8, cand_size=8)
                scores = model(x=imgs)
                loss = criterion(scores, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            scores = model(x=imgs)
            loss = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
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
    for imgs, labels in tqdm(loader, desc="Validating"): # Added tqdm
        imgs, labels = imgs.to(device), labels.to(device)
        # scores = model(x=imgs, num_rows=3, num_cols=3, ctx_size=8, cand_size=8)
        scores = model(x=imgs)
        loss = criterion(scores, labels)
        preds = scores.argmax(dim=1)
        correct += (preds==labels).sum().item(); total += labels.size(0)
        total_loss += loss.item() * imgs.size(0)
    return total_loss/len(loader.dataset), correct/total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RandomRotate90:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        # Apply the rotation using the efficient functional API
        return transforms.functional.rotate(img, angle)