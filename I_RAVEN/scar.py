# Adapted from https://github.com/mikomel/sal

from data_utility.data_utility_alb import dataset 
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar during training and validation
import albumentations as A
import argparse
import json
import math
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


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


class OddOneOutSCAR(SCAR):
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
        x = self.sal(x, in_rows=3, in_cols=3)
        x = self.global_model(x)
        x = x.view(B, P, -1)
        return x
    
    

def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss=0; total_correct=0; total_samples = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        scores = model(x=imgs, num_rows=3, num_cols=3, ctx_size=8, cand_size=8)
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
        scores = model(x=imgs, num_rows=3, num_cols=3, ctx_size=8, cand_size=8)
        loss = criterion(scores, labels)
        preds = scores.argmax(dim=1)
        correct += (preds==labels).sum().item(); total += labels.size(0)
        total_loss += loss.item() * imgs.size(0)
    return total_loss/len(loader.dataset), correct/total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class RandomRotate90:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(img, angle)

def main():
    RAVEN_ROOT_DIR = "/vol/bitbucket/icc24/RAVEN/data10k/"
    #fix seed for reproducibility
    set_seed(42)
    torch.autograd.set_detect_anomaly(True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    BETA_1 = 0.9
    BETA_2 = 0.999
    EPSILON = 1e-8

    IMG_SIZE = 80
    PATIENCE = 10

    writer = SummaryWriter(log_dir='runs/RAVEN_SCAR')

    # --- DataLoaders ---
    from types import SimpleNamespace
    args = SimpleNamespace(
        path=RAVEN_ROOT_DIR,
        img_size=IMG_SIZE,
        percent=100,
    )

    transform = [
        A.VerticalFlip(p=0.25),
        A.HorizontalFlip(p=0.25),
        A.RandomRotate90(p=0.25),
        A.Rotate(p=0.25),
        A.Transpose(p=0.25),
    ]

    train_dataset = dataset(args, mode="train", rpm_types=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'], transform=transform, transform_p=0.5)
    # train_dataset = dataset(args, mode="train", rpm_types=['cs'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)

    val_dataset = dataset(args, mode="val", rpm_types=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])
    # val_dataset = dataset(args, mode="val", rpm_types=['cs'])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    test_dataset = dataset(args, mode="test", rpm_types=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])
    # test_dataset = dataset(args, mode="test", rpm_types=['cs'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
    

    model = RAVENSCAR().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA_1, BETA_2),
        eps=EPSILON
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )

    best_val_acc = 0.0
    epochs_without_improvement = 0

    print(f"Using device: {DEVICE}")
    print(f"Model total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}:")
        # Reset memory stats if using GPU
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        train_time = time.time() - start_time

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Train Time: {train_time:.2f} seconds")
        
        mem_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
        mem_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
        print(f"  Mem Allocated: {mem_allocated:.2f} MB")
        print(f"  Mem Reserved: {mem_reserved:.2f} MB")

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        if scheduler:
            scheduler.step(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Mem/allocated", mem_allocated, epoch)
        writer.add_scalar("Mem/reserved", mem_reserved, epoch)
        writer.add_scalar("Time/epoch", train_time, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "./saved_models/best_model_scar.pth")
            print(f"Best model saved at epoch {epoch} with validation accuracy {val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"No improvement in validation accuracy for {PATIENCE} epochs. Stopping training.")
                break

    # Save the model state
    torch.save(model.state_dict(), "./saved_models/model_final_scar.pth")
    print("Training complete. Model saved as 'model_final_scar.pth'.")

    print("Training finished.")

    # --- Testing ---
    print("Testing the model on test dataset...")
    model.load_state_dict(torch.load("./saved_models/best_model_scar.pth"))
    model.eval()
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    writer.add_scalar("Loss/test", test_loss, EPOCHS)
    writer.add_scalar("Acc/test", test_acc, EPOCHS)


if __name__ == "__main__":
    main()