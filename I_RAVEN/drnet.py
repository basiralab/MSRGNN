# -*- coding:utf-8 -*-
# Adapted from code by 'Vecchio' at https://github.com/VecchioID/DRNet

from data_utility.data_utility import dataset
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from PIL import Image
from tensorboardX import SummaryWriter
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import argparse


DR_S, DR_F = .1, .5  # Dropout prob. for spatial and fully-connected layers.
O_HC, O_OC = 64, 64  # Hidden and output channels for original enc.
F_HC, F_OC = 64, 16  # Hidden and output channels for frame enc.
S_HC, S_OC = 128, 64  # Hidden and output channels for sequence enc.
F_PL, S_PL = 5 * 5, 16  # Pooled sizes for frame and sequence enc. outputs.
F_Z = F_OC * F_PL  # Frame embedding dimensions.
K_D = 7  # Conv. kernel dimensions.

BL_IN = 3
BLOUT = F_Z
G_IN = BLOUT
G_HID = G_IN
G_OUT = G_IN
R_OUT = 32
C_DIM = 2
P_DIM = 32
C = 1.0

class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.name = args.save_model_name

    def load_model(self, path, epoch):
        state_dict = torch.load(path + '{}_epoch_{}.pth'.format(self.name, epoch))['state_dict']
        self.load_state_dict(state_dict)

    def save_model(self, path, epoch, acc, loss):
        torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss},
                   path + '{}_epoch_{}_acc_{}.pth'.format(self.name, epoch, acc))

    def compute_loss(self, output, target):
        pass

    def train_(self, image, target):
        self.optimizer.zero_grad()
        output = self(image)
        loss = self.compute_loss(output, target)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, image, target):
        with torch.no_grad():
            output = self(image)
        loss = self.compute_loss(output, target)
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def test_(self, image, target):
        with torch.no_grad():
            output = self(image)
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return accuracy


class perm(nn.Module):
    def __init__(self):
        super(perm, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class flat(nn.Module):
    def __init__(self):
        super(flat, self).__init__()

    def forward(self, x):
        return x.flatten(1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim):
        super(ConvBlock, self).__init__()
        self.conv = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, K_D, stride=dim, padding=K_D // 2)
        self.bnrm = getattr(nn, 'BatchNorm{}d'.format(dim))(out_ch)
        self.drop = nn.Sequential(perm(), nn.Dropout2d(DR_S), perm()) if dim == 1 else nn.Dropout2d(DR_S)
        self.block = nn.Sequential(self.conv, nn.ReLU(), self.bnrm, self.drop)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, hd_ch, out_ch, dim):
        super(ResBlock, self).__init__()
        self.dim = dim
        self.conv = nn.Sequential(ConvBlock(in_ch, hd_ch, dim), ConvBlock(hd_ch, out_ch, dim))
        self.down = nn.Sequential(nn.MaxPool2d(3, 2, 1), nn.MaxPool2d(3, 2, 1))
        self.skip = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.conv(x) + self.skip(x if self.dim == 1 else self.down(x))


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model
        self.stack = lambda x: torch.stack([torch.cat((x[:, :8], x[:, i].unsqueeze(1)), dim=1) for i in range(8, 16)],
                                           dim=1)

        self.obj_enc = ViT(in_channels=1, patch_size=20, emb_size=400, img_size=80, depth=16, n_classes=1000)
        self.obj_enc_conv = nn.Sequential(ResBlock(1, F_HC, F_HC, 2), ResBlock(F_HC, F_HC, F_OC, 2))
        self.seq_enc = nn.Sequential(ResBlock(9, S_OC, S_HC, 1), nn.MaxPool1d(6, 4, 1), ResBlock(S_HC, S_HC, S_OC, 1),
                                     nn.AdaptiveAvgPool1d(S_PL))

        self.linear = nn.Sequential(nn.Linear(1024, 512), nn.ELU(), nn.BatchNorm1d(512), nn.Dropout(DR_F),
                                    nn.Linear(512, 256), nn.ELU(), nn.BatchNorm1d(256), nn.Dropout(DR_F),
                                    nn.Linear(256, 8 if model == 'Context-blind' else 1))

        self.linear_proj_feature = nn.Linear(2, 1)

    def forward(self, x):
        x = x.view(-1, 1, 80, 80)
        x1 = self.obj_enc(x).flatten(1)
        x2 = self.obj_enc_conv(x).flatten(1)
        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1)), dim=1).permute(0, 2, 1)
        x = self.linear_proj_feature(x).squeeze(2)
        x = x.view(-1, 16, F_Z)
        x = self.stack(x)
        x = self.seq_enc(x.view(-1, 9, F_Z)).flatten(1)
        return self.linear(x).view(-1, 8)


class Solver(BasicModel):
    def __init__(self, args):
        super(Solver, self).__init__(args)
        self.model = args.model
        self.net = nn.DataParallel(Net(args.model), device_ids=[0, 1]) if args.multi_gpu else Net(args.model)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def compute_loss(self, output, target):
        return F.cross_entropy(output, target)

    def forward(self, x):
        x = 1 - x / 255.0
        out = self.net(x)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), )
        # nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
        
# ---------- TRAIN & VALIDATION ----------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss=0; total_correct=0; total_samples = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        scores = model(imgs)
        loss = criterion(scores, labels)
        loss.backward()
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
        scores = model(imgs)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()

    RAVEN_ROOT_DIR = args.data
    #fix seed for reproducibility
    set_seed(42)
    torch.autograd.set_detect_anomaly(True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 100 
    IMG_SIZE = 80
    PATIENCE = 10
    LEARNING_RATE = 3e-4
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8

    writer = SummaryWriter(log_dir='runs/RAVEN_DRNET')

    # --- DataLoaders ---
    from types import SimpleNamespace
    args = SimpleNamespace(
        path=RAVEN_ROOT_DIR,
        img_size=IMG_SIZE,
        dataset="raven",
        percent=100,
    )

    transform = None

    train_dataset = dataset(args, mode="train", rpm_types=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'], transform=transform, transform_p=0.0)
    # train_dataset = dataset(args, mode="train", rpm_types=['cs'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)

    val_dataset = dataset(args, mode="val", rpm_types=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])
    # val_dataset = dataset(args, mode="val", rpm_types=['cs'])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    test_dataset = dataset(args, mode="test", rpm_types=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])
    # test_dataset = dataset(args, mode="test", rpm_types=['cs'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    model_args = SimpleNamespace(
        model='vit',
        epochs=401,
        load_workers=4,
        dataset_path='',
        save_model_name='',
        img_size=IMG_SIZE,
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        dataset="raven",
        multi_gpu=False,
        val_every=5,
        test_every=5,
        percent=100,
        # For nargs='+', argparse returns a list, so we mimic that behavior
        trn_configs=['*'],
        tst_configs=['*'],
        silent=False,
        shuffle_first=False,
        check_point=False
    )

    model = Solver(model_args).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=EPSILON)
    criterion = nn.CrossEntropyLoss()
    scheduler = None

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
            scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Mem/allocated", mem_allocated, epoch)
        writer.add_scalar("Mem/reserved", mem_reserved, epoch)
        writer.add_scalar("Time/epoch", train_time, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "./saved_models/best_model_drnet.pth")
            print(f"Best model saved at epoch {epoch} with validation accuracy {val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"No improvement in validation accuracy for {PATIENCE} epochs. Stopping training.")
                break

    # Save the model state
    torch.save(model.state_dict(), "./saved_models/model_final_drnet.pth")
    print("Training complete. Model saved as 'model_final_drnet.pth'.")

    print("Training finished.")

    # --- Testing ---
    print("Testing the model on test dataset...")

    model.load_state_dict(torch.load("./saved_models/best_model_drnet.pth"))
    model.eval()
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    writer.add_scalar("Loss/test", test_loss, EPOCHS)
    writer.add_scalar("Acc/test", test_acc, EPOCHS)


if __name__ == "__main__":
    main()