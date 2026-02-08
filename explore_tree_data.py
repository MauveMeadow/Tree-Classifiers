#!/usr/bin/env python3
"""
TreeSatAI GeoJSON Data Exploration and CNN Classification Preparation

This script explores the GeoJSON data for tree species classification.
Each GeoJSON file represents a different tree species with Sentinel-2 satellite features.

Data generated from Google Earth Engine with:
- Sentinel-2 imagery from Germany (March-October 2022)
- 8 monthly composites (temporal dimension)
- 5x5 spatial patches via neighborhoodToArray (spatial dimension)
- 10 spectral bands + 5 vegetation indices per timestep

Features include:
- Sentinel-2 bands (B2-B12, B8A) across 8 months: 10 bands × 8 timesteps
- Vegetation indices (NDVI, EVI, EVI2, NDWI, SAVI) across 8 months: 5 indices × 8 timesteps
- Each feature is a 5×5 array (spatial patch from 2-pixel kernel)
- Coordinates (X, Y) in UTM projection
- Labels: l1_leaf_types (broadleaf/needleleaf), l2_genus, l3_species

Data shape for CNN: (samples, channels, height, width, timesteps)
where channels = 15 (10 bands + 5 indices), height = width = 5, timesteps = 8
"""

import json
import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# =============================================
# CONFIGURATION
# =============================================
DATA_DIR = Path('/home/ubuntu/TreeSatAI/')
GEOJSON_PATTERN = '*.geojson'

# Sentinel-2 bands used (B1, B9 excluded, matching GEE script Bands variable)
SENTINEL_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
TIME_STEPS = 8  # March (base) + April-October (_1 through _7)

# Vegetation indices calculated in GEE
VEG_INDICES = ['NDVI', 'EVI', 'EVI2', 'SAVI', 'NDWI']

# Spatial patch size (from ee.Kernel.rectangle(2, 2) -> 5x5 window)
PATCH_SIZE = 5

# Label columns (hierarchical classification)
LABEL_COLS = ['l1_leaf_types', 'l2_genus', 'l3_species']

# Tree hierarchy from GEE script
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

# =============================================
# DATA LOADING FUNCTIONS
# =============================================

def load_geojson(filepath):
    """Load a single GeoJSON file and return features."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['features']


def parse_filename(filename):
    """Parse tree classification info from filename."""
    name = filename.replace('.geojson', '')
    parts = name.split('_')
    
    leaf_type = parts[0]
    genus = parts[1] if len(parts) > 1 else 'unknown'
    species = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
    
    return {
        'leaf_type': leaf_type,
        'genus': genus, 
        'species': species,
        'full_name': name
    }


def get_feature_names():
    """Get ordered list of spectral feature names matching GEE export order."""
    features = []
    
    # For each timestep (base + _1 through _7)
    for t in range(TIME_STEPS):
        suffix = '' if t == 0 else f'_{t}'
        
        # Sentinel-2 bands
        for band in SENTINEL_BANDS:
            features.append(f"{band}{suffix}")
        
        # Vegetation indices
        for idx in VEG_INDICES:
            features.append(f"{idx}{suffix}")
    
    return features


def load_all_data(data_dir):
    """Load all GeoJSON files and combine into a single dataset."""
    files = sorted(glob.glob(str(data_dir / GEOJSON_PATTERN)))
    
    all_features = []
    file_stats = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        tree_info = parse_filename(filename)
        
        features = load_geojson(filepath)
        
        for feat in features:
            props = feat['properties'].copy()
            props['source_file'] = filename
            props['parsed_leaf_type'] = tree_info['leaf_type']
            props['parsed_genus'] = tree_info['genus']
            props['parsed_species'] = tree_info['species']
            all_features.append(props)
        
        file_stats.append({
            'filename': filename,
            'num_samples': len(features),
            **tree_info
        })
    
    return all_features, file_stats


# =============================================
# DATA EXPLORATION FUNCTIONS
# =============================================

def analyze_dataset_structure(file_stats):
    """Print dataset structure summary."""
    print("=" * 70)
    print("DATASET STRUCTURE SUMMARY")
    print("=" * 70)
    
    df_stats = pd.DataFrame(file_stats)
    
    print(f"\nTotal files: {len(file_stats)}")
    print(f"Total samples: {df_stats['num_samples'].sum():,}")
    
    print("\n--- Samples per file ---")
    for _, row in df_stats.iterrows():
        print(f"  {row['filename']}: {row['num_samples']:,} samples")
    
    print("\n--- Class Distribution ---")
    print("\nBy Leaf Type (L1):")
    for leaf_type in df_stats['leaf_type'].unique():
        subset = df_stats[df_stats['leaf_type'] == leaf_type]
        total = subset['num_samples'].sum()
        print(f"  {leaf_type}: {total:,} samples ({len(subset)} species)")
    
    print("\nBy Genus (L2):")
    for genus in sorted(df_stats['genus'].unique()):
        subset = df_stats[df_stats['genus'] == genus]
        total = subset['num_samples'].sum()
        print(f"  {genus}: {total:,} samples ({len(subset)} species)")
    
    return df_stats


def analyze_feature_availability(all_features):
    """Analyze which features have actual data vs null values."""
    print("\n" + "=" * 70)
    print("FEATURE AVAILABILITY ANALYSIS")
    print("=" * 70)
    
    if not all_features:
        print("No features found!")
        return None, None
    
    sample_props = all_features[0]
    all_keys = list(sample_props.keys())
    
    expected_features = get_feature_names()
    
    print(f"\nExpected spectral features: {len(expected_features)}")
    print(f"  {len(SENTINEL_BANDS)} bands × {TIME_STEPS} timesteps = {len(SENTINEL_BANDS) * TIME_STEPS}")
    print(f"  {len(VEG_INDICES)} indices × {TIME_STEPS} timesteps = {len(VEG_INDICES) * TIME_STEPS}")
    
    non_null_counts = defaultdict(int)
    array_features = defaultdict(list)
    total_samples = len(all_features)
    
    for feat in all_features:
        for key, value in feat.items():
            if value is not None:
                non_null_counts[key] += 1
                if isinstance(value, list):
                    if len(array_features[key]) < 3:
                        array_features[key].append(value)
    
    always_available = []
    partially_available = []
    always_null = []
    
    for key in all_keys:
        pct = non_null_counts[key] / total_samples * 100
        if pct >= 99.9:
            always_available.append((key, pct))
        elif pct > 0:
            partially_available.append((key, pct))
        else:
            always_null.append(key)
    
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Total property columns: {len(all_keys)}")
    
    print(f"\n--- Always Available ({len(always_available)}) ---")
    for key, pct in sorted(always_available):
        array_info = ""
        if key in array_features and array_features[key]:
            sample = array_features[key][0]
            if isinstance(sample, list) and len(sample) > 0:
                if isinstance(sample[0], list):
                    array_info = f" [2D: {len(sample)}×{len(sample[0])}]"
                else:
                    array_info = f" [1D: {len(sample)}]"
        print(f"  {key}: {pct:.1f}%{array_info}")
    
    print(f"\n--- Partially Available ({len(partially_available)}) ---")
    for key, pct in sorted(partially_available, key=lambda x: -x[1])[:15]:
        array_info = ""
        if key in array_features and array_features[key]:
            sample = array_features[key][0]
            if isinstance(sample, list) and len(sample) > 0:
                if isinstance(sample[0], list):
                    array_info = f" [2D: {len(sample)}×{len(sample[0])}]"
                else:
                    array_info = f" [1D: {len(sample)}]"
        print(f"  {key}: {pct:.1f}%{array_info}")
    if len(partially_available) > 15:
        print(f"  ... and {len(partially_available) - 15} more")
    
    print(f"\n--- Always Null ({len(always_null)}) ---")
    if always_null:
        band_groups = defaultdict(list)
        for key in always_null:
            base = key.split('_')[0]
            band_groups[base].append(key)
        for base in sorted(band_groups.keys()):
            print(f"  {base}: {len(band_groups[base])} columns")
    else:
        print("  None - all features have data!")
    
    return non_null_counts, array_features


def analyze_label_distribution(all_features):
    """Analyze label distribution for classification."""
    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    df = pd.DataFrame(all_features)
    
    for level, col in enumerate(['l1_leaf_types', 'l2_genus', 'l3_species'], 1):
        print(f"\n--- Level {level}: {col} ---")
        value_counts = df[col].value_counts()
        for label, count in value_counts.items():
            pct = count / len(df) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    print("\n--- Class Imbalance Metrics ---")
    for col in ['l1_leaf_types', 'l2_genus', 'l3_species']:
        counts = df[col].value_counts()
        imbalance_ratio = counts.max() / counts.min()
        print(f"  {col}: {len(counts)} classes, imbalance ratio = {imbalance_ratio:.2f}")
    
    return df


def analyze_coordinates(all_features):
    """Analyze spatial distribution of samples."""
    print("\n" + "=" * 70)
    print("SPATIAL DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    coords = [(f['X'], f['Y']) for f in all_features if f.get('X') and f.get('Y')]
    
    if not coords:
        print("No coordinate data available!")
        return
    
    x_vals = [c[0] for c in coords]
    y_vals = [c[1] for c in coords]
    
    print(f"\nTotal samples with coordinates: {len(coords):,}")
    print(f"\nX coordinate (Easting):")
    print(f"  Min: {min(x_vals):.2f}, Max: {max(x_vals):.2f}")
    print(f"  Range: {max(x_vals) - min(x_vals):.2f}")
    print(f"\nY coordinate (Northing):")
    print(f"  Min: {min(y_vals):.2f}, Max: {max(y_vals):.2f}")
    print(f"  Range: {max(y_vals) - min(y_vals):.2f}")


# =============================================
# DATA PREPARATION FOR CNN
# =============================================

def prepare_dataset_for_cnn(all_features, label_column='l3_species'):
    """
    Prepare dataset for CNN training with 5×5 spatial patches.
    
    Expected data shape from GEE neighborhoodToArray with kernel.rectangle(2,2):
    - Each spectral feature is a 5×5 array
    - 15 channels (10 bands + 5 indices) × 8 timesteps = 120 feature arrays
    
    Output shape: (samples, channels, height, width, timesteps)
    For 2D CNN: (samples, channels × timesteps, height, width) = (N, 120, 5, 5)
    """
    print("\n" + "=" * 70)
    print("CNN DATA PREPARATION")
    print("=" * 70)
    
    feature_names = get_feature_names()
    num_channels = len(SENTINEL_BANDS) + len(VEG_INDICES)
    
    print(f"\nExpected data structure:")
    print(f"  Channels: {num_channels} (10 S2 bands + 5 indices)")
    print(f"  Timesteps: {TIME_STEPS} (March-October)")
    print(f"  Spatial patch: {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"  Total features: {len(feature_names)}")
    
    # Analyze data type
    sample = all_features[0]
    sample_val = sample.get('B2')
    
    if sample_val is None:
        print("\nChecking for array data...")
        has_arrays = False
        for feat in all_features[:100]:
            for fn in feature_names:
                val = feat.get(fn)
                if val is not None and isinstance(val, list):
                    has_arrays = True
                    print(f"  Found array data in {fn}: shape {np.array(val).shape}")
                    break
            if has_arrays:
                break
        
        if not has_arrays:
            print("\nWARNING: No spectral data found in samples!")
            print("The GeoJSON may have empty spectral values.")
            return None, None, None, None
    
    # Check if data is array or scalar
    is_array_data = isinstance(sample_val, list) if sample_val else False
    
    if is_array_data:
        return _prepare_array_data(all_features, label_column, feature_names, num_channels)
    else:
        return _prepare_scalar_data(all_features, label_column, feature_names, num_channels)


def _prepare_array_data(all_features, label_column, feature_names, num_channels):
    """Prepare data when features are 5×5 arrays (from neighborhoodToArray)."""
    print("\nProcessing array data (5×5 patches)...")
    
    X_list = []
    y_list = []
    
    for feat in all_features:
        sample_arrays = []
        label = feat.get(label_column, 'unknown')
        valid = True
        
        for fn in feature_names:
            val = feat.get(fn)
            if val is not None and isinstance(val, list):
                arr = np.array(val, dtype=np.float32)
                if arr.ndim == 1:
                    side = int(np.sqrt(len(arr)))
                    arr = arr.reshape(side, side) if side * side == len(arr) else None
                sample_arrays.append(arr)
            else:
                sample_arrays.append(None)
                valid = False
        
        if valid:
            X_list.append(sample_arrays)
            y_list.append(label)
    
    if not X_list:
        print("No complete samples found!")
        return None, None, None, None
    
    # Convert to numpy: (samples, features, H, W)
    n_samples = len(X_list)
    n_features = len(feature_names)
    
    # Infer patch size from first sample
    patch_h, patch_w = X_list[0][0].shape
    
    X = np.zeros((n_samples, n_features, patch_h, patch_w), dtype=np.float32)
    for i, sample_arrays in enumerate(X_list):
        for j, arr in enumerate(sample_arrays):
            if arr is not None:
                X[i, j] = arr
    
    # Reshape to (samples, channels, H, W, timesteps)
    X = X.reshape(n_samples, TIME_STEPS, num_channels, patch_h, patch_w)
    X = X.transpose(0, 2, 3, 4, 1)  # (N, C, H, W, T)
    
    y = np.array(y_list)
    
    print(f"\nFinal shape: {X.shape} (samples, channels, height, width, timesteps)")
    
    # Label encoding
    unique_labels = np.unique(y)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    y_encoded = np.array([label_to_idx[l] for l in y])
    
    print(f"Classes: {len(unique_labels)}")
    
    return X, y_encoded, label_to_idx, idx_to_label


def _prepare_scalar_data(all_features, label_column, feature_names, num_channels):
    """Prepare data when features are scalars (single pixel values)."""
    print("\nProcessing scalar data...")
    
    X_list = []
    y_list = []
    
    for feat in all_features:
        values = []
        label = feat.get(label_column, 'unknown')
        has_data = False
        
        for fn in feature_names:
            val = feat.get(fn)
            if val is not None and not isinstance(val, list):
                values.append(float(val))
                has_data = True
            else:
                values.append(np.nan)
        
        if has_data:
            X_list.append(values)
            y_list.append(label)
    
    if not X_list:
        print("No samples with scalar data found!")
        return None, None, None, None
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    
    # Reshape for 1D CNN: (samples, channels, timesteps)
    X = X.reshape(-1, TIME_STEPS, num_channels)
    X = X.transpose(0, 2, 1)  # (N, C, T)
    
    print(f"\nFinal shape: {X.shape} (samples, channels, timesteps)")
    
    # Handle NaN
    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"NaN values: {nan_mask.sum()} ({100*nan_mask.mean():.2f}%)")
        for c in range(X.shape[1]):
            channel_mean = np.nanmean(X[:, c, :])
            X[:, c, :] = np.where(np.isnan(X[:, c, :]), channel_mean, X[:, c, :])
    
    # Label encoding
    unique_labels = np.unique(y)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    y_encoded = np.array([label_to_idx[l] for l in y])
    
    print(f"Classes: {len(unique_labels)}")
    
    return X, y_encoded, label_to_idx, idx_to_label


def prepare_flat_features(all_features, label_column='l3_species'):
    """Prepare flattened features for MLP/Random Forest."""
    print("\n" + "=" * 70)
    print("FLAT FEATURE EXTRACTION")
    print("=" * 70)
    
    X_list = []
    y_list = []
    feature_names = get_feature_names()
    
    for feat in all_features:
        features = [feat.get('X', 0), feat.get('Y', 0)]
        
        for fn in feature_names:
            val = feat.get(fn)
            if val is not None:
                if isinstance(val, list):
                    features.extend(np.array(val).flatten().tolist())
                else:
                    features.append(val)
            else:
                features.append(0)
        
        X_list.append(features)
        y_list.append(feat.get(label_column, 'unknown'))
    
    max_len = max(len(f) for f in X_list)
    X_padded = [f + [0] * (max_len - len(f)) for f in X_list]
    
    X = np.array(X_padded, dtype=np.float32)
    y = np.array(y_list)
    
    # Normalize
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1
    X = (X - mean) / std
    X = np.nan_to_num(X, 0)
    
    # Label encoding
    unique_labels = np.unique(y)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[l] for l in y])
    
    print(f"Samples: {len(X):,}, Features: {X.shape[1]}")
    print(f"Classes: {len(unique_labels)}")
    
    return X, y_encoded, label_to_idx


# =============================================
# CNN MODEL ARCHITECTURE (PyTorch)
# =============================================

def get_cnn_model_code():
    """Return example PyTorch CNN model code."""
    return '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class TreeClassifier2DCNN(nn.Module):
    """
    2D CNN for tree classification from multi-temporal spatial patches.
    
    Input: (batch, channels, height, width, timesteps)
    Reshape to: (batch, channels*timesteps, height, width) for 2D conv
    """
    
    def __init__(self, num_channels=15, patch_size=5, num_timesteps=8, num_classes=19, dropout=0.4):
        super().__init__()
        
        in_channels = num_channels * num_timesteps  # 15 * 8 = 120
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (batch, channels, H, W, timesteps)
        batch, C, H, W, T = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(batch, C*T, H, W)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).flatten(1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class TreeClassifier1DCNN(nn.Module):
    """
    1D CNN for temporal classification (when no spatial patches).
    Input: (batch, channels, timesteps)
    """
    
    def __init__(self, num_channels=15, num_timesteps=8, num_classes=19, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).flatten(1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class TreeClassifier3DCNN(nn.Module):
    """
    3D CNN processing spatial AND temporal dimensions together.
    Input: (batch, channels, height, width, timesteps)
    """
    
    def __init__(self, num_channels=15, patch_size=5, num_timesteps=8, num_classes=19, dropout=0.4):
        super().__init__()
        
        # 3D convolutions: (C, D, H, W) where D=timesteps
        self.conv1 = nn.Conv3d(num_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: (batch, C, H, W, T) -> need (batch, C, T, H, W) for Conv3d
        x = x.permute(0, 1, 4, 2, 3)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).flatten(1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def compute_class_weights(y):
    """Compute class weights for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return torch.FloatTensor(weights)


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, 
                class_weights=None, device='cpu', patience=15):
    """Train with early stopping and learning rate scheduling."""
    
    model = model.to(device)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(1 - val_acc/100)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_tree_classifier.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    model.load_state_dict(torch.load('best_tree_classifier.pth'))
    return model


# Training example:
"""
from sklearn.model_selection import train_test_split

# Load data (assuming X and y from prepare_dataset_for_cnn)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train)

# DataLoaders
train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# Model selection based on data shape
if X.ndim == 5:  # (N, C, H, W, T)
    model = TreeClassifier2DCNN(num_classes=len(np.unique(y)))
    # or model = TreeClassifier3DCNN(num_classes=len(np.unique(y)))
else:  # (N, C, T)
    model = TreeClassifier1DCNN(num_classes=len(np.unique(y)))

# Class weights for imbalanced data
weights = compute_class_weights(y_train)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = train_model(model, train_loader, val_loader, epochs=100, 
                    class_weights=weights, device=device)

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = []
    for X_batch, _ in test_loader:
        outputs = model(X_batch.to(device))
        y_pred.extend(outputs.argmax(1).cpu().numpy())

print(classification_report(y_test, y_pred, target_names=list(idx_to_label.values())))
"""
'''


# =============================================
# MAIN EXECUTION
# =============================================

def main():
    """Main exploration pipeline."""
    
    print("=" * 70)
    print("TreeSatAI GeoJSON Data Exploration")
    print("=" * 70)
    print(f"\nData directory: {DATA_DIR}")
    
    # Load all data
    print("\nLoading data...")
    all_features, file_stats = load_all_data(DATA_DIR)
    
    if not all_features:
        print("ERROR: No data loaded!")
        return None, None, None
    
    # Analysis
    df_stats = analyze_dataset_structure(file_stats)
    non_null_counts, array_features = analyze_feature_availability(all_features)
    df = analyze_label_distribution(all_features)
    analyze_coordinates(all_features)
    
    # Prepare for CNN
    result = prepare_dataset_for_cnn(all_features)
    
    if result[0] is not None:
        X, y, label_to_idx, idx_to_label = result
        print(f"\n✓ CNN data ready: {X.shape}")
        
        # Save processed arrays
        np.save(DATA_DIR / 'X_features.npy', X)
        np.save(DATA_DIR / 'y_labels.npy', y)
        with open(DATA_DIR / 'label_mapping.json', 'w') as f:
            json.dump({str(k): v for k, v in idx_to_label.items()}, f, indent=2)
        print(f"Saved: X_features.npy, y_labels.npy, label_mapping.json")
    else:
        print("\n⚠ No CNN-ready spectral data - using coordinates only")
        X, y, label_to_idx = prepare_flat_features(all_features)
    
    # Print model code
    print("\n" + "=" * 70)
    print("PYTORCH CNN MODEL CODE")
    print("=" * 70)
    print(get_cnn_model_code())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Dataset: {len(all_features):,} samples across {len(file_stats)} tree species
Labels: 
  - L1 (leaf type): 2 classes (broadleaf, needleleaf)
  - L2 (genus): {len(df['l2_genus'].unique())} classes
  - L3 (species): {len(df['l3_species'].unique())} classes

Recommended approaches:
1. Hierarchical classification (L1 → L2 → L3)
2. Use class weights to handle imbalance
3. Data augmentation (flips, rotations for 2D patches)
4. Cross-validation for robust evaluation
""")
    
    # Save summary
    df_stats.to_csv(DATA_DIR / 'dataset_summary.csv', index=False)
    
    return df, all_features, file_stats


if __name__ == "__main__":
    df, all_features, file_stats = main()
