from data_utility.data_utility_norm import dataset
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import numpy as np
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

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
        return x.view(-1, 16, self.out_channels*4*4)
    
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
        return x.view(-1, 8, self.output_dim)
    
class PanelEmbedding(nn.Module):
    def __init__(self, tag=True):
        super(PanelEmbedding, self).__init__()
        self.in_dim = 512 + (9 if tag else 0)
        self.fc = nn.Linear(self.in_dim, 256)

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
        self.mlp = MLP()
        self.proj = PanelEmbedding(tag=use_tag)
        self.tags = self.tag_panels()


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
    
def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss=0; total_correct = 0; total_samples = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        scores = model(imgs)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

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
    LEARNING_RATE = 1e-4
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8

    writer = SummaryWriter(log_dir='runs/RAVEN_WREN')

    # --- DataLoaders ---
    from types import SimpleNamespace
    args = SimpleNamespace(
        path=RAVEN_ROOT_DIR,
        img_size=IMG_SIZE,
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

    model = WReN(image_size=IMG_SIZE).to(DEVICE)
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
            torch.save(model.state_dict(), "./saved_models/best_model_wren.pth")
            print(f"Best model saved at epoch {epoch} with validation accuracy {val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"No improvement in validation accuracy for {PATIENCE} epochs. Stopping training.")
                break

    # Save the model state
    torch.save(model.state_dict(), "./saved_models/model_final_wren.pth")
    print("Training complete. Model saved as 'model_final_wren.pth'.")

    print("Training finished.")

    # --- Testing ---
    print("Testing the model on test dataset...")

    model.load_state_dict(torch.load("./saved_models/best_model_wren.pth"))
    model.eval()
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    writer.add_scalar("Loss/test", test_loss, EPOCHS)
    writer.add_scalar("Acc/test", test_acc, EPOCHS)


if __name__ == "__main__":
    main()