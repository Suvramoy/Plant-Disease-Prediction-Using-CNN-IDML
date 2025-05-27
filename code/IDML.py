# =============== Import Libraries ===============
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
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from PIL import Image
import sys
import argparse
from tqdm import tqdm
import wandb

# Add temp110 to Python Path
sys.path.append("/kaggle/input/temp110")

# Import metric learning functions from PyTorch Metric Learning
try:
    from pytorch_metric_learning import losses
except ImportError:
    print("pytorch-metric-learning not found in temp110. Install it with '!pip install pytorch-metric-learning' or verify temp110 contents.")
    sys.exit(1)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# =============== Argument Parser ===============
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='IDML Implementation with Proxy-Anchor Loss for Cassava Leaf Disease Classification')
    parser.add_argument('--dataset_path', default='/kaggle/input/cassava-leaf-disease-classification/', help='Path to dataset')
    parser.add_argument('--log_dir', default='/kaggle/working/logs', help='Path to log folder')
    parser.add_argument('--embedding_size', default=512, type=int, help='Size of embedding')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of samples per batch')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--workers', default=2, type=int, help='Number of workers for dataloader')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--lr_decay_step', default=5, type=int, help='Learning rate decay step')
    parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help='Learning rate decay gamma')
    parser.add_argument('--alpha', default=32, type=float, help='Scaling parameter for Proxy-Anchor')
    parser.add_argument('--mrg', default=0.1, type=float, help='Margin parameter for Proxy-Anchor')
    parser.add_argument('--warm', default=1, type=int, help='Warmup training epochs')
    parser.add_argument('--bn_freeze', default=1, type=int, help='Batch normalization freeze')
    parser.add_argument('--l2_norm', default=1, type=int, help='L2 normalization')
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
LOG_DIR = os.path.join(args.log_dir, f'cassava_resnet50_proxyanchor_embedding{args.embedding_size}_alpha{args.alpha}_mrg{args.mrg}_lr{args.lr}_batch{args.batch_size}')
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_PATH = os.path.join(LOG_DIR, 'saved_models')
os.makedirs(MODEL_PATH, exist_ok=True)

wandb.init(project='cassava_proxyanchor', notes=LOG_DIR, mode='offline')
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

# =============== IDML Model ===============
class IDMLModel(nn.Module):
    def __init__(self, num_classes=nb_classes, embedding_size=args.embedding_size):
        super(IDMLModel, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc_semantic = nn.Linear(2048, embedding_size)
        self.fc_uncertainty = nn.Linear(2048, embedding_size)
        if args.l2_norm:
            self.norm = nn.functional.normalize

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, start_dim=1)
        semantic_embedding = self.fc_semantic(features)
        uncertainty_embedding = self.fc_uncertainty(features)
        if args.l2_norm:
            semantic_embedding = self.norm(semantic_embedding, p=2, dim=1)
            uncertainty_embedding = self.norm(uncertainty_embedding, p=2, dim=1)
        return semantic_embedding, uncertainty_embedding

# =============== Evaluation Metrics ===============
def compute_metrics(embeddings, labels, proxies, num_classes, k=2):
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    # Detach proxies before converting to NumPy
    proxies_norm = torch.nn.functional.normalize(proxies, p=2, dim=1).detach().cpu().numpy()

    # Compute similarity scores (cosine similarity) between embeddings and proxies
    scores = embeddings @ proxies_norm.T  # Shape: (N, num_classes)
    sorted_indices = np.argsort(scores, axis=1)[:, ::-1]  # Sort in descending order

    # R@1 (same as accuracy)
    top1_preds = sorted_indices[:, 0]
    r_at_1 = np.mean(top1_preds == labels)

    # R@2
    top2_preds = sorted_indices[:, :2]
    r_at_2 = np.mean([labels[i] in top2_preds[i] for i in range(len(labels))])

    # NMI: Cluster embeddings using K-means and compute NMI with true labels
    kmeans = KMeans(n_clusters=num_classes, random_state=args.seed).fit(embeddings)
    cluster_labels = kmeans.labels_
    nmi = normalized_mutual_info_score(labels, cluster_labels)

    # RP (Precision@R): For each sample, check if top-R predictions contain the true label (R=k)
    precision_at_r = np.mean([labels[i] in sorted_indices[i, :k] for i in range(len(labels))])

    # MC@R (Mean Class Recall@R): Compute recall per class, then average
    class_recalls = []
    for c in range(num_classes):
        class_mask = labels == c
        if class_mask.sum() == 0:
            continue
        class_top_k = sorted_indices[class_mask, :k]
        class_recall = np.mean([labels[i] in class_top_k[j] for j, i in enumerate(np.where(class_mask)[0])])
        class_recalls.append(class_recall)
    mc_at_r = np.mean(class_recalls) if class_recalls else 0.0

    return {
        'R@1': r_at_1,
        'R@2': r_at_2,
        'NMI': nmi,
        'RP': precision_at_r,
        'MC@R': mc_at_r
    }

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
            for m in model.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        if args.warm > 0:
            if epoch == 0:
                for param in model.feature_extractor.parameters():
                    param.requires_grad = False
            if epoch == args.warm:
                for param in model.feature_extractor.parameters():
                    param.requires_grad = True

        total_train_loss = 0
        correct_train = 0
        total_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            semantic_emb, uncertainty_emb = model(images)
            loss = criterion(semantic_emb, labels)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            torch.nn.utils.clip_grad_value_(criterion.proxies, 10)
            optimizer.step()

            total_train_loss += loss.item()
            proxies_norm = torch.nn.functional.normalize(criterion.proxies, p=2, dim=1)
            preds = torch.argmax(torch.mm(semantic_emb, proxies_norm.t()), dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            pbar.set_postfix({'Loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation Phase with Metrics
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        all_embeddings = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                semantic_emb, _ = model(images)
                val_loss = criterion(semantic_emb, labels)
                total_val_loss += val_loss.item()
                preds = torch.argmax(torch.mm(semantic_emb, proxies_norm.t()), dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_embeddings.append(semantic_emb)
                all_labels.append(labels)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # Compute Metrics
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_embeddings, all_labels, criterion.proxies, nb_classes, k=2)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        print(f"Metrics - R@1: {metrics['R@1']:.4f}, R@2: {metrics['R@2']:.4f}, NMI: {metrics['NMI']:.4f}, RP: {metrics['RP']:.4f}, MC@R: {metrics['MC@R']:.4f}")

        wandb.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'R@1': metrics['R@1'],
            'R@2': metrics['R@2'],
            'NMI': metrics['NMI'],
            'RP': metrics['RP'],
            'MC@R': metrics['MC@R']
        }, step=epoch)

        # Save model every epoch for debugging
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
    model = IDMLModel(num_classes=nb_classes, embedding_size=args.embedding_size).to(device)
    criterion = losses.ProxyAnchorLoss(num_classes=nb_classes, embedding_size=args.embedding_size, margin=args.mrg, alpha=args.alpha).to(device)
    
    param_groups = [
        {'params': list(set(model.feature_extractor.parameters()).difference(set(model.fc_semantic.parameters()) | set(model.fc_uncertainty.parameters())))},
        {'params': list(model.fc_semantic.parameters()) + list(model.fc_uncertainty.parameters()), 'lr': args.lr * 1},
        {'params': criterion.proxies, 'lr': args.lr * 100}
    ]
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    print("Training parameters:", vars(args))
    print(f"Training for {args.epochs} epochs.")
    print("🚀 Training Model...")
    train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # Final Evaluation with Metrics
    checkpoint = torch.load(os.path.join(MODEL_PATH, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            semantic_emb, _ = model(images)
            all_embeddings.append(semantic_emb)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    final_metrics = compute_metrics(all_embeddings, all_labels, criterion.proxies, nb_classes, k=2)
    print(f"🎯 Final Metrics - R@1: {final_metrics['R@1']:.4f}, R@2: {final_metrics['R@2']:.4f}, NMI: {final_metrics['NMI']:.4f}, RP: {final_metrics['RP']:.4f}, MC@R: {final_metrics['MC@R']:.4f}")