#!/usr/bin/env python3
"""
TreeSatAI - Hierarchical Tree Species Classification with CNN

This script implements a hierarchical classification approach:
- L1: Leaf type (broadleaf vs needleleaf) - 2 classes
- L2: Genus (beech, oak, pine, etc.) - 9 classes  
- L3: Species (european beech, scots pine, etc.) - 19 classes

Features:
1. Hierarchical classification (L1 → L2 → L3)
2. Class weights for imbalanced data
3. Data augmentation (flips, rotations for 5×5 patches)
4. K-fold cross-validation
5. Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
"""

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, balanced_accuracy_score,
    top_k_accuracy_score
)
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
DATA_DIR = Path('/home/ubuntu/TreeSatAI/')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 15
NUM_FOLDS = 5

# Data shape: (samples, channels, height, width, timesteps)
NUM_CHANNELS = 15
PATCH_SIZE = 5
NUM_TIMESTEPS = 8

print(f"Using device: {DEVICE}")
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================
# HIERARCHICAL LABEL MAPPING
# =============================================
TREE_HIERARCHY = {
    'broadleaf': {
        'beech': ['european beech'],
        'oak': ['sessile oak', 'english oak', 'red oak'],
        'long-lived deciduous': ['sycamore maple', 'european ash', 'linden', 'cherry'],
        'short-lived deciduous': ['alder', 'poplar', 'birch']
    },
    'needleleaf': {
        'fir': ['silver fir'],
        'larch': ['european larch', 'japanese larch'],
        'spruce': ['norway spruce'],
        'pine': ['scots pine', 'black pine', 'weymouth pine'],
        'douglas fir': ['douglas fir']
    }
}

def build_label_mappings():
    """Build hierarchical label mappings."""
    l1_classes = list(TREE_HIERARCHY.keys())
    l2_classes = []
    l3_classes = []
    
    species_to_l1 = {}
    species_to_l2 = {}
    l2_to_l1 = {}
    
    for l1, l2_dict in TREE_HIERARCHY.items():
        for l2, species_list in l2_dict.items():
            if l2 not in l2_classes:
                l2_classes.append(l2)
            l2_to_l1[l2] = l1
            for species in species_list:
                l3_classes.append(species)
                species_to_l1[species] = l1
                species_to_l2[species] = l2
    
    return {
        'l1_classes': l1_classes,
        'l2_classes': l2_classes,
        'l3_classes': l3_classes,
        'species_to_l1': species_to_l1,
        'species_to_l2': species_to_l2,
        'l2_to_l1': l2_to_l1,
        'l1_to_idx': {c: i for i, c in enumerate(l1_classes)},
        'l2_to_idx': {c: i for i, c in enumerate(l2_classes)},
        'l3_to_idx': {c: i for i, c in enumerate(l3_classes)},
        'idx_to_l1': {i: c for i, c in enumerate(l1_classes)},
        'idx_to_l2': {i: c for i, c in enumerate(l2_classes)},
        'idx_to_l3': {i: c for i, c in enumerate(l3_classes)},
    }


# =============================================
# DATA AUGMENTATION
# =============================================
class TreeDataset(Dataset):
    """Dataset with augmentation for 5×5 spatial patches."""
    
    def __init__(self, X, y_l1, y_l2, y_l3, augment=False):
        self.X = torch.FloatTensor(X)
        self.y_l1 = torch.LongTensor(y_l1)
        self.y_l2 = torch.LongTensor(y_l2)
        self.y_l3 = torch.LongTensor(y_l3)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # (C, H, W, T)
        
        if self.augment:
            x = self._augment(x)
        
        return x, self.y_l1[idx], self.y_l2[idx], self.y_l3[idx]
    
    def _augment(self, x):
        """Apply random augmentations to spatial patches."""
        # x shape: (C, H, W, T)
        
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[2])  # Flip W dimension
        
        # Random vertical flip
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])  # Flip H dimension
        
        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
        
        # Random temporal shift (circular)
        if torch.rand(1) > 0.5:
            shift = torch.randint(1, x.shape[3], (1,)).item()
            x = torch.roll(x, shifts=shift, dims=3)
        
        # Small noise injection
        if torch.rand(1) > 0.7:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        return x


# =============================================
# CNN MODELS
# =============================================
class HierarchicalTreeCNN(nn.Module):
    """
    Hierarchical CNN with shared backbone and separate heads for L1, L2, L3.
    
    Architecture:
    - Shared 2D CNN backbone for spatial-temporal features
    - L1 head: broadleaf vs needleleaf
    - L2 head: genus classification (conditioned on L1)
    - L3 head: species classification (conditioned on L2)
    """
    
    def __init__(self, num_channels=15, patch_size=5, num_timesteps=8,
                 num_l1=2, num_l2=9, num_l3=19, dropout=0.4):
        super().__init__()
        
        in_channels = num_channels * num_timesteps  # 15 * 8 = 120
        
        # Shared backbone
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        
        # Shared feature layer
        self.fc_shared = nn.Linear(512, 256)
        
        # L1 head (leaf type)
        self.fc_l1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_l1)
        )
        
        # L2 head (genus) - receives L1 features
        self.fc_l2 = nn.Sequential(
            nn.Linear(256 + num_l1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_l2)
        )
        
        # L3 head (species) - receives L1 and L2 features
        self.fc_l3 = nn.Sequential(
            nn.Linear(256 + num_l1 + num_l2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_l3)
        )
    
    def forward(self, x, return_features=False):
        # x: (batch, C, H, W, T)
        batch, C, H, W, T = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(batch, C*T, H, W)
        
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        
        # Shared features
        features = F.relu(self.fc_shared(x))
        features = self.dropout(features)
        
        # L1 prediction
        out_l1 = self.fc_l1(features)
        l1_probs = F.softmax(out_l1, dim=1)
        
        # L2 prediction (conditioned on L1)
        l2_input = torch.cat([features, l1_probs], dim=1)
        out_l2 = self.fc_l2(l2_input)
        l2_probs = F.softmax(out_l2, dim=1)
        
        # L3 prediction (conditioned on L1 and L2)
        l3_input = torch.cat([features, l1_probs, l2_probs], dim=1)
        out_l3 = self.fc_l3(l3_input)
        
        if return_features:
            return out_l1, out_l2, out_l3, features
        
        return out_l1, out_l2, out_l3


class TreeCNN3D(nn.Module):
    """
    3D CNN that processes spatial AND temporal dimensions together.
    Alternative architecture for comparison.
    """
    
    def __init__(self, num_channels=15, num_l1=2, num_l2=9, num_l3=19, dropout=0.4):
        super().__init__()
        
        # 3D convolutions: (C, T, H, W)
        self.conv1 = nn.Conv3d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        
        self.fc_shared = nn.Linear(128, 64)
        
        self.fc_l1 = nn.Linear(64, num_l1)
        self.fc_l2 = nn.Linear(64 + num_l1, num_l2)
        self.fc_l3 = nn.Linear(64 + num_l1 + num_l2, num_l3)
    
    def forward(self, x):
        # x: (batch, C, H, W, T) -> (batch, C, T, H, W)
        x = x.permute(0, 1, 4, 2, 3)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        
        features = F.relu(self.fc_shared(x))
        
        out_l1 = self.fc_l1(features)
        l1_probs = F.softmax(out_l1, dim=1)
        
        out_l2 = self.fc_l2(torch.cat([features, l1_probs], dim=1))
        l2_probs = F.softmax(out_l2, dim=1)
        
        out_l3 = self.fc_l3(torch.cat([features, l1_probs, l2_probs], dim=1))
        
        return out_l1, out_l2, out_l3


# =============================================
# LOSS FUNCTION
# =============================================
class HierarchicalLoss(nn.Module):
    """
    Combined loss for hierarchical classification.
    
    Total loss = α * L1_loss + β * L2_loss + γ * L3_loss
    
    With optional consistency penalty for hierarchical violations.
    """
    
    def __init__(self, weights_l1=None, weights_l2=None, weights_l3=None,
                 alpha=1.0, beta=1.0, gamma=1.0, consistency_weight=0.1):
        super().__init__()
        
        self.criterion_l1 = nn.CrossEntropyLoss(weight=weights_l1)
        self.criterion_l2 = nn.CrossEntropyLoss(weight=weights_l2)
        self.criterion_l3 = nn.CrossEntropyLoss(weight=weights_l3)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.consistency_weight = consistency_weight
    
    def forward(self, out_l1, out_l2, out_l3, y_l1, y_l2, y_l3):
        loss_l1 = self.criterion_l1(out_l1, y_l1)
        loss_l2 = self.criterion_l2(out_l2, y_l2)
        loss_l3 = self.criterion_l3(out_l3, y_l3)
        
        total_loss = self.alpha * loss_l1 + self.beta * loss_l2 + self.gamma * loss_l3
        
        return total_loss, loss_l1, loss_l2, loss_l3


# =============================================
# TRAINING FUNCTIONS
# =============================================
def compute_class_weights(y, num_classes):
    """Compute inverse frequency class weights."""
    counts = np.bincount(y, minlength=num_classes)
    counts = np.maximum(counts, 1)  # Avoid division by zero
    weights = len(y) / (num_classes * counts)
    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    losses = {'l1': 0, 'l2': 0, 'l3': 0}
    
    for X_batch, y_l1, y_l2, y_l3 in loader:
        X_batch = X_batch.to(device)
        y_l1, y_l2, y_l3 = y_l1.to(device), y_l2.to(device), y_l3.to(device)
        
        optimizer.zero_grad()
        out_l1, out_l2, out_l3 = model(X_batch)
        
        loss, loss_l1, loss_l2, loss_l3 = criterion(out_l1, out_l2, out_l3, y_l1, y_l2, y_l3)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        losses['l1'] += loss_l1.item()
        losses['l2'] += loss_l2.item()
        losses['l3'] += loss_l3.item()
    
    n = len(loader)
    return total_loss / n, {k: v/n for k, v in losses.items()}


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    
    all_preds = {'l1': [], 'l2': [], 'l3': []}
    all_labels = {'l1': [], 'l2': [], 'l3': []}
    all_probs = {'l1': [], 'l2': [], 'l3': []}
    
    with torch.no_grad():
        for X_batch, y_l1, y_l2, y_l3 in loader:
            X_batch = X_batch.to(device)
            y_l1, y_l2, y_l3 = y_l1.to(device), y_l2.to(device), y_l3.to(device)
            
            out_l1, out_l2, out_l3 = model(X_batch)
            
            loss, _, _, _ = criterion(out_l1, out_l2, out_l3, y_l1, y_l2, y_l3)
            total_loss += loss.item()
            
            # Predictions
            all_preds['l1'].extend(out_l1.argmax(1).cpu().numpy())
            all_preds['l2'].extend(out_l2.argmax(1).cpu().numpy())
            all_preds['l3'].extend(out_l3.argmax(1).cpu().numpy())
            
            # Labels
            all_labels['l1'].extend(y_l1.cpu().numpy())
            all_labels['l2'].extend(y_l2.cpu().numpy())
            all_labels['l3'].extend(y_l3.cpu().numpy())
            
            # Probabilities
            all_probs['l1'].extend(F.softmax(out_l1, dim=1).cpu().numpy())
            all_probs['l2'].extend(F.softmax(out_l2, dim=1).cpu().numpy())
            all_probs['l3'].extend(F.softmax(out_l3, dim=1).cpu().numpy())
    
    # Convert to numpy
    for key in all_preds:
        all_preds[key] = np.array(all_preds[key])
        all_labels[key] = np.array(all_labels[key])
        all_probs[key] = np.array(all_probs[key])
    
    return total_loss / len(loader), all_preds, all_labels, all_probs


def compute_metrics(y_true, y_pred, y_prob, num_classes, level_name):
    """Compute comprehensive metrics for a classification level."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Top-k accuracy (for L3 with many classes)
    if num_classes > 5:
        try:
            metrics['top3_accuracy'] = top_k_accuracy_score(y_true, y_prob, k=3, labels=range(num_classes))
            metrics['top5_accuracy'] = top_k_accuracy_score(y_true, y_prob, k=5, labels=range(num_classes))
        except:
            pass
    
    return metrics


# =============================================
# VISUALIZATION
# =============================================
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(max(10, len(class_names)), max(8, len(class_names)*0.8)))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # L1 Accuracy
    ax = axes[0, 1]
    ax.plot(history['train_acc_l1'], label='Train')
    ax.plot(history['val_acc_l1'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('L1 (Leaf Type) Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # L2 Accuracy
    ax = axes[1, 0]
    ax.plot(history['train_acc_l2'], label='Train')
    ax.plot(history['val_acc_l2'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('L2 (Genus) Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # L3 Accuracy
    ax = axes[1, 1]
    ax.plot(history['train_acc_l3'], label='Train')
    ax.plot(history['val_acc_l3'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('L3 (Species) Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics_per_class, class_names, level_name, save_path):
    """Plot per-class precision, recall, F1."""
    fig, ax = plt.subplots(figsize=(max(12, len(class_names)*0.6), 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, metrics_per_class['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_per_class['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_per_class['f1'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f'{level_name} Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================
# CROSS-VALIDATION
# =============================================
def run_cross_validation(X, y_l1, y_l2, y_l3, label_mappings, n_folds=5):
    """Run stratified k-fold cross-validation."""
    
    print(f"\n{'='*70}")
    print(f"RUNNING {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    cv_results = {
        'l1': defaultdict(list),
        'l2': defaultdict(list),
        'l3': defaultdict(list)
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_l3)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_l1_train, y_l1_val = y_l1[train_idx], y_l1[val_idx]
        y_l2_train, y_l2_val = y_l2[train_idx], y_l2[val_idx]
        y_l3_train, y_l3_val = y_l3[train_idx], y_l3[val_idx]
        
        # Compute class weights
        weights_l1 = compute_class_weights(y_l1_train, len(label_mappings['l1_classes'])).to(DEVICE)
        weights_l2 = compute_class_weights(y_l2_train, len(label_mappings['l2_classes'])).to(DEVICE)
        weights_l3 = compute_class_weights(y_l3_train, len(label_mappings['l3_classes'])).to(DEVICE)
        
        # Create datasets
        train_dataset = TreeDataset(X_train, y_l1_train, y_l2_train, y_l3_train, augment=True)
        val_dataset = TreeDataset(X_val, y_l1_val, y_l2_val, y_l3_val, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Model
        model = HierarchicalTreeCNN(
            num_channels=NUM_CHANNELS,
            patch_size=PATCH_SIZE,
            num_timesteps=NUM_TIMESTEPS,
            num_l1=len(label_mappings['l1_classes']),
            num_l2=len(label_mappings['l2_classes']),
            num_l3=len(label_mappings['l3_classes'])
        ).to(DEVICE)
        
        # Loss and optimizer
        criterion = HierarchicalLoss(weights_l1, weights_l2, weights_l3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            train_loss, _ = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, DEVICE)
            
            val_acc_l3 = accuracy_score(val_labels['l3'], val_preds['l3'])
            scheduler.step(val_loss)
            
            if val_acc_l3 > best_val_acc:
                best_val_acc = val_acc_l3
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 20 == 0:
                val_acc_l1 = accuracy_score(val_labels['l1'], val_preds['l1'])
                val_acc_l2 = accuracy_score(val_labels['l2'], val_preds['l2'])
                print(f"  Epoch {epoch+1}: L1={val_acc_l1:.3f}, L2={val_acc_l2:.3f}, L3={val_acc_l3:.3f}")
        
        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        _, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, DEVICE)
        
        # Compute metrics for each level
        for level, num_classes in [('l1', 2), ('l2', 9), ('l3', 19)]:
            metrics = compute_metrics(
                val_labels[level], val_preds[level], val_probs[level],
                num_classes, level
            )
            for metric_name, value in metrics.items():
                cv_results[level][metric_name].append(value)
        
        print(f"  Fold {fold+1} Results:")
        print(f"    L1 Accuracy: {cv_results['l1']['accuracy'][-1]:.4f}")
        print(f"    L2 Accuracy: {cv_results['l2']['accuracy'][-1]:.4f}")
        print(f"    L3 Accuracy: {cv_results['l3']['accuracy'][-1]:.4f}")
    
    return cv_results


# =============================================
# MAIN TRAINING
# =============================================
def train_final_model(X_train, X_test, y_l1_train, y_l1_test, y_l2_train, y_l2_test,
                      y_l3_train, y_l3_test, label_mappings):
    """Train final model with train/test split."""
    
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*70}")
    
    # Further split train into train/val
    X_train, X_val, y_l1_train, y_l1_val, y_l2_train, y_l2_val, y_l3_train, y_l3_val = \
        train_test_split(X_train, y_l1_train, y_l2_train, y_l3_train, 
                         test_size=0.15, stratify=y_l3_train, random_state=SEED)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Class weights
    weights_l1 = compute_class_weights(y_l1_train, len(label_mappings['l1_classes'])).to(DEVICE)
    weights_l2 = compute_class_weights(y_l2_train, len(label_mappings['l2_classes'])).to(DEVICE)
    weights_l3 = compute_class_weights(y_l3_train, len(label_mappings['l3_classes'])).to(DEVICE)
    
    # Datasets
    train_dataset = TreeDataset(X_train, y_l1_train, y_l2_train, y_l3_train, augment=True)
    val_dataset = TreeDataset(X_val, y_l1_val, y_l2_val, y_l3_val, augment=False)
    test_dataset = TreeDataset(X_test, y_l1_test, y_l2_test, y_l3_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = HierarchicalTreeCNN(
        num_channels=NUM_CHANNELS,
        patch_size=PATCH_SIZE,
        num_timesteps=NUM_TIMESTEPS,
        num_l1=len(label_mappings['l1_classes']),
        num_l2=len(label_mappings['l2_classes']),
        num_l3=len(label_mappings['l3_classes'])
    ).to(DEVICE)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = HierarchicalLoss(weights_l1, weights_l2, weights_l3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Training history
    history = defaultdict(list)
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    print("\nTraining...")
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_losses = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Evaluate on train and val
        _, train_preds, train_labels, _ = evaluate(model, train_loader, criterion, DEVICE)
        val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, DEVICE)
        
        # Compute accuracies
        train_acc_l1 = accuracy_score(train_labels['l1'], train_preds['l1'])
        train_acc_l2 = accuracy_score(train_labels['l2'], train_preds['l2'])
        train_acc_l3 = accuracy_score(train_labels['l3'], train_preds['l3'])
        
        val_acc_l1 = accuracy_score(val_labels['l1'], val_preds['l1'])
        val_acc_l2 = accuracy_score(val_labels['l2'], val_preds['l2'])
        val_acc_l3 = accuracy_score(val_labels['l3'], val_preds['l3'])
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc_l1'].append(train_acc_l1)
        history['train_acc_l2'].append(train_acc_l2)
        history['train_acc_l3'].append(train_acc_l3)
        history['val_acc_l1'].append(val_acc_l1)
        history['val_acc_l2'].append(val_acc_l2)
        history['val_acc_l3'].append(val_acc_l3)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc_l3 > best_val_acc:
            best_val_acc = val_acc_l3
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}/{val_loss:.4f} | "
                  f"L1={train_acc_l1:.3f}/{val_acc_l1:.3f} | "
                  f"L2={train_acc_l2:.3f}/{val_acc_l2:.3f} | "
                  f"L3={train_acc_l3:.3f}/{val_acc_l3:.3f}")
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'label_mappings': label_mappings,
        'history': dict(history),
    }, DATA_DIR / 'tree_classifier_model.pth')
    print(f"\nModel saved to {DATA_DIR / 'tree_classifier_model.pth'}")
    
    # Plot training history
    plot_training_history(history, DATA_DIR / 'training_history.png')
    print(f"Training history saved to {DATA_DIR / 'training_history.png'}")
    
    return model, history, test_loader, criterion


def evaluate_final_model(model, test_loader, criterion, label_mappings):
    """Final evaluation on test set with all metrics."""
    
    print(f"\n{'='*70}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*70}")
    
    _, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion, DEVICE)
    
    results = {}
    
    for level, (num_classes, class_names) in [
        ('l1', (2, label_mappings['l1_classes'])),
        ('l2', (9, label_mappings['l2_classes'])),
        ('l3', (19, label_mappings['l3_classes']))
    ]:
        print(f"\n--- {level.upper()} ({['Leaf Type', 'Genus', 'Species'][['l1','l2','l3'].index(level)]}) ---")
        
        # Compute metrics
        metrics = compute_metrics(
            test_labels[level], test_preds[level], test_probs[level],
            num_classes, level
        )
        results[level] = metrics
        
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
        
        if 'top3_accuracy' in metrics:
            print(f"  Top-3 Accuracy:    {metrics['top3_accuracy']:.4f}")
            print(f"  Top-5 Accuracy:    {metrics['top5_accuracy']:.4f}")
        
        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(test_labels[level], test_preds[level], 
                                    target_names=class_names, zero_division=0))
        
        # Confusion matrix
        plot_confusion_matrix(
            test_labels[level], test_preds[level], class_names,
            f'{level.upper()} Confusion Matrix',
            DATA_DIR / f'confusion_matrix_{level}.png'
        )
        
        # Per-class metrics
        precision_per_class = precision_score(test_labels[level], test_preds[level], 
                                              average=None, zero_division=0)
        recall_per_class = recall_score(test_labels[level], test_preds[level], 
                                        average=None, zero_division=0)
        f1_per_class = f1_score(test_labels[level], test_preds[level], 
                                average=None, zero_division=0)
        
        plot_per_class_metrics(
            {'precision': precision_per_class, 'recall': recall_per_class, 'f1': f1_per_class},
            class_names, level.upper(),
            DATA_DIR / f'per_class_metrics_{level}.png'
        )
    
    print(f"\nConfusion matrices saved to {DATA_DIR}")
    print(f"Per-class metric plots saved to {DATA_DIR}")
    
    return results


# =============================================
# MAIN
# =============================================
def main():
    """Main training pipeline."""
    
    print("="*70)
    print("TreeSatAI - Hierarchical Tree Species Classification")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    X = np.load(DATA_DIR / 'X_features.npy')
    y = np.load(DATA_DIR / 'y_labels.npy')
    
    with open(DATA_DIR / 'label_mapping.json') as f:
        idx_to_species = json.load(f)
    idx_to_species = {int(k): v for k, v in idx_to_species.items()}
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {len(idx_to_species)}")
    
    # Build hierarchical mappings
    label_mappings = build_label_mappings()
    
    # Map species indices to hierarchical labels
    species_to_l3_idx = {s: i for i, s in enumerate(label_mappings['l3_classes'])}
    
    y_l1 = np.array([label_mappings['l1_to_idx'][label_mappings['species_to_l1'][idx_to_species[i]]] for i in y])
    y_l2 = np.array([label_mappings['l2_to_idx'][label_mappings['species_to_l2'][idx_to_species[i]]] for i in y])
    y_l3 = np.array([species_to_l3_idx[idx_to_species[i]] for i in y])
    
    print(f"\nHierarchical labels:")
    print(f"  L1 classes: {label_mappings['l1_classes']}")
    print(f"  L2 classes: {label_mappings['l2_classes']}")
    print(f"  L3 classes: {len(label_mappings['l3_classes'])} species")
    
    # Train/test split (stratified by L3)
    X_train, X_test, y_l1_train, y_l1_test, y_l2_train, y_l2_test, y_l3_train, y_l3_test = \
        train_test_split(X, y_l1, y_l2, y_l3, test_size=0.2, stratify=y_l3, random_state=SEED)
    
    print(f"\nData split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Run cross-validation
    cv_results = run_cross_validation(X_train, y_l1_train, y_l2_train, y_l3_train, 
                                      label_mappings, n_folds=NUM_FOLDS)
    
    # Print CV summary
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    for level in ['l1', 'l2', 'l3']:
        level_name = ['Leaf Type', 'Genus', 'Species'][['l1','l2','l3'].index(level)]
        print(f"\n{level.upper()} ({level_name}):")
        for metric in ['accuracy', 'balanced_accuracy', 'f1_macro']:
            values = cv_results[level][metric]
            print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Train final model
    model, history, test_loader, criterion = train_final_model(
        X_train, X_test, y_l1_train, y_l1_test, y_l2_train, y_l2_test,
        y_l3_train, y_l3_test, label_mappings
    )
    
    # Final evaluation
    results = evaluate_final_model(model, test_loader, criterion, label_mappings)
    
    # Save results
    with open(DATA_DIR / 'evaluation_results.json', 'w') as f:
        json.dump({level: {k: float(v) for k, v in metrics.items()} 
                   for level, metrics in results.items()}, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nSaved files:")
    print(f"  - tree_classifier_model.pth (trained model)")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix_l1/l2/l3.png")
    print(f"  - per_class_metrics_l1/l2/l3.png")
    print(f"  - evaluation_results.json")
    
    return model, results, cv_results


if __name__ == "__main__":
    model, results, cv_results = main()
