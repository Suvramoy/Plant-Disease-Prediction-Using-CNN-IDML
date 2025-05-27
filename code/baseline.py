import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import sys
import argparse
from tqdm import tqdm
import wandb

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# =============== Argument Parser ===============
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Baseline ResNet Model for Cassava Leaf Disease Classification')
    parser.add_argument('--dataset_path', default='/kaggle/input/cassava-leaf-disease-classification/', help='Path to dataset')
    parser.add_argument('--log_dir', default='/kaggle/working/logs', help='Path to log folder')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of samples per batch')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--workers', default=2, type=int, help='Number of workers for dataloader')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--lr_decay_step', default=5, type=int, help='Learning rate decay step')
    parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help='Learning rate decay gamma')
    parser.add_argument('--warm', default=1, type=int, help='Warmup training epochs')
    parser.add_argument('--bn_freeze', default=1, type=int, help='Batch normalization freeze')
    parser.add_argument('--early_stopping_patience', default=6, type=int, help='Patience for early stopping')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')

    if argv is None:
        filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith('-f') and '.json' not in arg]
        args = parser.parse_args(filtered_args)
    else:
        args = parser.parse_args(argv)
    return args

args = parse_args()

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directory setup
LOG_DIR = os.path.join(args.log_dir, f'cassava_resnet50_baseline_lr{args.lr}_batch{args.batch_size}')
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_PATH = os.path.join(LOG_DIR, 'saved_models')
os.makedirs(MODEL_PATH, exist_ok=True)

wandb.init(project='cassava_baseline', notes=LOG_DIR, mode='offline')
wandb.config.update(args)

# =============== Dataset Preparation ===============
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['img_path']).convert("RGB")
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

train_df = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
train_df['img_path'] = train_df['image_id'].apply(lambda x: os.path.join(args.dataset_path, 'train_images', x))

train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=args.seed)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CassavaDataset(train_data, transform=train_transforms)
val_dataset = CassavaDataset(val_data, transform=val_transforms)

# Weighted sampling
class_counts = train_df['label'].value_counts().sort_index()
weights = 1. / class_counts[train_data['label']].values
sampler = WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

nb_classes = train_df['label'].nunique()
print(f"Number of classes: {nb_classes}")
print("Class distribution:\n", train_df['label'].value_counts(normalize=True))

# =============== Baseline ResNet Model ===============
class BaselineResNet(nn.Module):
    def __init__(self, num_classes=nb_classes):
        super(BaselineResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# =============== Training Loop ===============
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=args.epochs, patience=args.early_stopping_patience):
    model.to(device)
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        if args.bn_freeze:
            for m in model.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        if args.warm > 0:
            if epoch == 0:
                for param in model.model.parameters():
                    param.requires_grad = False
                for param in model.model.fc.parameters():
                    param.requires_grad = True
            if epoch == args.warm:
                for param in model.model.parameters():
                    param.requires_grad = True

        total_train_loss = 0
        correct_train = 0
        total_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            optimizer.step()

            total_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            pbar.set_postfix({'Loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        wandb.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }, step=epoch)

        # Save model every epoch
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(MODEL_PATH, f'model_epoch_{epoch+1}.pth'))
        print(f"🛠️ Model saved at {os.path.join(MODEL_PATH, f'model_epoch_{epoch+1}.pth')}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(MODEL_PATH, 'best_model.pth'))
            print("✅ Model saved (New best validation loss)!")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"⏳ Early stopping triggered after {patience} epochs of no improvement.")
            break

        scheduler.step()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'training_plots.png'))
    plt.show()

    return train_losses, val_losses, train_accs, val_accs

# =============== Main Execution ===============
if __name__ == "__main__":
    model = BaselineResNet(num_classes=nb_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    print("Training parameters:", vars(args))
    print(f"Training for {args.epochs} epochs.")
    print("🚀 Training Model...")
    train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # Final Evaluation
    checkpoint = torch.load(os.path.join(MODEL_PATH, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    final_accuracy = correct / total
    print(f"🎯 Final Validation Accuracy: {final_accuracy:.4f}")