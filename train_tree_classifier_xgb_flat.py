#!/usr/bin/env python3
"""
Author: Mays Alsheikh
TreeSatAI - Flat Hierarchical Tree Species Classification with XGBoost

This script implements a standard flattened multiclass approach using the 
same "ultimate" feature set and parameters as the Top-Down model for comparison.
"""

import numpy as np
import netCDF4 as nc
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import time
import warnings
import multiprocessing as mp
from collections import Counter
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
DATA_DIR = Path('data/')
S2_NC = DATA_DIR / 's2_bands_indices_gradients_3x3_seasonal.nc'
DEM_NC = DATA_DIR / 'TreeSatAI_DEM_3x3.nc'
SEED = 42
NUM_FOLDS = 5
AVAILABLE_GPUS = [0, 1, 2, 4, 5, 6, 7]
FAST_MODE = False
AUGMENT = True
NOISE_STD = 0.01
CHECKPOINT_DIR = Path('checkpoints_flat')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Same winning bands from the Top-Down experiments
WINNING_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
FINAL_INDICES = ["NDVI", "EVI", "EVI2", "SAVI", "NDWI", 
                 "NDVI_DIFF", "EVI_DIFF", "EVI2_DIFF", "SAVI_DIFF", "NDWI_DIFF"]
FINAL_FEATS = WINNING_BANDS + ["MSK"] + FINAL_INDICES
FINAL_MONTHS = list(range(1, 13))

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


def get_mappings():
    species_to_path = {}
    for l1, l2_dict in TREE_HIERARCHY.items():
        for l2, species_list in l2_dict.items():
            for s in species_list:
                species_to_path[s] = {'l1': l1, 'l2': l2, 'l3': s}
    return species_to_path


def get_feature_indices(feature_keys, months):
    BAND_MAP = {'B1': 0, 'B2': 1, 'B3': 2, 'B4': 3, 'B5': 4, 'B6': 5,
                'B7': 6, 'B8': 7, 'B8A': 8, 'B9': 9, 'B11': 10, 'B12': 11, 'MSK': 12}
    INDEX_START = {'NDVI': 156, 'EVI': 168, 'EVI2': 180, 'SAVI': 192, 'NDWI': 204}
    DIFF_START = {'NDVI_DIFF': 216, 'EVI_DIFF': 227, 'EVI2_DIFF': 238, 'SAVI_DIFF': 249, 'NDWI_DIFF': 260}
    
    selected_indices = []
    for m in months:
        m_idx = m - 1
        for feat in feature_keys:
            if feat in BAND_MAP:
                selected_indices.append(m_idx * 13 + BAND_MAP[feat])
            elif feat in INDEX_START:
                selected_indices.append(INDEX_START[feat] + m_idx)
            elif feat in DIFF_START and m_idx > 0:
                selected_indices.append(DIFF_START[feat] + m_idx - 1)
    return sorted(list(set(selected_indices)))


def load_base_data():
    species_path_map = get_mappings()
    with nc.Dataset(S2_NC, 'r') as ds_s2:
        num_samples_s2 = ds_s2.variables['label'].shape[0]
    with nc.Dataset(DEM_NC, 'r') as ds_dem:
        num_samples_dem = ds_dem.variables['patches'].shape[0]
    num_samples = min(num_samples_s2, num_samples_dem)
    with nc.Dataset(S2_NC, 'r') as ds:
        labels_raw = ds.variables['label'][:num_samples]
    y_df = []
    for label in labels_raw:
        if isinstance(label, bytes): label = label.decode('utf-8')
        path = species_path_map.get(label, species_path_map['european beech'])
        y_df.append(path)
    return num_samples, y_df


def get_feature_matrix(num_samples, selected_indices):
    with nc.Dataset(S2_NC, 'r') as ds_s2, nc.Dataset(DEM_NC, 'r') as ds_dem:
        s2_var = 'X' if 'X' in ds_s2.variables else 'feature'
        X_s2 = ds_s2.variables[s2_var][:num_samples, :, :, selected_indices]
        X_dem = ds_dem.variables['patches'][:num_samples, :, :, :]
    X_s2 = np.transpose(X_s2, (0, 3, 1, 2)).reshape(num_samples, -1)
    X_dem = np.transpose(X_dem, (0, 3, 1, 2)).reshape(num_samples, -1)
    X = np.concatenate([X_s2, X_dem], axis=1)
    X[X < -500] = 0
    return X


def oversample_and_augment(X_in, y_in):
    l3_labels = [row['l3'] for row in y_in]
    counts = Counter(l3_labels)
    max_count = max(counts.values())
    X_extra, y_extra = [], []
    for label, count in counts.items():
        if count < max_count:
            num_to_add = max_count - count
            class_indices = [i for i, l in enumerate(l3_labels) if l == label]
            extra_indices = np.random.choice(class_indices, num_to_add, replace=True)
            X_sub = X_in[extra_indices].copy()
            y_sub = [y_in[i] for i in extra_indices]
            X_sub = X_sub + np.random.normal(0, NOISE_STD, X_sub.shape)
            X_extra.append(X_sub)
            y_extra.extend(y_sub)
    if X_extra:
        return np.concatenate([X_in] + X_extra, axis=0), y_in + y_extra
    return X_in, y_in


def run_single_fold(args):
    fold, train_idx, val_idx, X, y_df, gpu_id, extra_params, name = args
    print(f"\n--- FOLD {fold+1} Starting on GPU {gpu_id} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_struct = [y_df[i] for i in train_idx]
    y_val_struct = [y_df[i] for i in val_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    scale_factor = extra_params.pop('_scale_data', 1)
    aug_all = extra_params.pop('_aug_all', False)
    spatial_aug = extra_params.pop('_spatial_aug', False)
    balance_power = extra_params.pop('_balance_power', 1.0)
    
    if AUGMENT:
        if scale_factor > 1 or aug_all or balance_power > 0 or spatial_aug:
            l3_labels = [row['l3'] for row in y_train_struct]
            counts = Counter(l3_labels)
            max_count = max(counts.values())
            X_extra, y_extra = [], []
            for label, count in counts.items():
                target_count = int(count * (max_count / count) ** balance_power)
                if target_count > count:
                    num_to_add = target_count - count
                    idx = [i for i, l in enumerate(l3_labels) if l == label]
                    extra_idx = np.random.choice(idx, num_to_add, replace=True)
                    X_extra.append(X_train[extra_idx])
                    y_extra.extend([y_train_struct[i] for i in extra_idx])
            if X_extra:
                X_train = np.concatenate([X_train] + X_extra, axis=0)
                y_train_struct = y_train_struct + y_extra
            if scale_factor > 1:
                X_orig = X_train.copy()
                y_orig = y_train_struct.copy()
                for _ in range(scale_factor - 1):
                    X_train = np.concatenate([X_train, X_orig], axis=0)
                    y_train_struct = y_train_struct + y_orig
            if spatial_aug:
                num_c = X_train.shape[1] // 9
                X_temp = X_train.reshape(-1, num_c, 3, 3)
                rots = np.random.randint(0, 4, size=len(X_temp))
                for k in range(1, 4):
                    idx = np.where(rots == k)[0]
                    if len(idx) > 0:
                        X_temp[idx] = np.rot90(X_temp[idx], k=k, axes=(2, 3))
                flip_h = np.random.random(len(X_temp)) > 0.5
                idx_h = np.where(flip_h)[0]
                if len(idx_h) > 0:
                    X_temp[idx_h] = np.flip(X_temp[idx_h], axis=3)
                flip_v = np.random.random(len(X_temp)) > 0.5
                idx_v = np.where(flip_v)[0]
                if len(idx_v) > 0:
                    X_temp[idx_v] = np.flip(X_temp[idx_v], axis=2)
                X_train = X_temp.reshape(len(X_temp), -1)
            if aug_all:
                X_train = X_train + np.random.normal(0, NOISE_STD, X_train.shape)
        else:
            X_train, y_train_struct = oversample_and_augment(X_train, y_train_struct)
    
    le = LabelEncoder()
    y_train = le.fit_transform([row['l3'] for row in y_train_struct])
    y_val = le.transform([row['l3'] for row in y_val_struct])
    g3 = [row['l3'] for row in y_val_struct]
    
    is_prod = not FAST_MODE
    xgb_defaults = {
        'n_estimators': 100,
        'max_depth': 6 if FAST_MODE else 0,
        'max_leaves': 1023,
        'grow_policy': 'lossguide' if is_prod else 'depthwise',
        'learning_rate': 0.1,
        'tree_method': 'hist',
        'device': f'cuda:{gpu_id}',
        'random_state': SEED,
        'n_jobs': 16,
        'early_stopping_rounds': 50 if is_prod else None
    }
    xgb_defaults.update(extra_params)
    
    clf = xgb.XGBClassifier(**xgb_defaults)
    if is_prod:
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1)
    else:
        clf.fit(X_train, y_train)
    
    safe_name = name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
    checkpoint_path = CHECKPOINT_DIR / f"{safe_name}_fold{fold+1}.json"
    clf.save_model(str(checkpoint_path))
    
    p3_idx = clf.predict(X_val)
    p3 = le.inverse_transform(p3_idx)
    
    species_path_map = get_mappings()
    p1 = [species_path_map[p]['l1'] for p in p3]
    p2 = [species_path_map[p]['l2'] for p in p3]
    g1 = [row['l1'] for row in y_val_struct]
    g2 = [row['l2'] for row in y_val_struct]
    
    res = {
        'l1_acc': accuracy_score(g1, p1),
        'l2_acc': accuracy_score(g2, p2),
        'l3_acc': accuracy_score(g3, p3),
        'l3_bal_acc': balanced_accuracy_score(g3, p3)
    }
    print(f"Fold {fold+1} Finished: L3 Acc: {res['l3_acc']:.4f}, L3 Bal Acc: {res['l3_bal_acc']:.4f}")
    return res

if __name__ == "__main__":
    num_samples, y_df = load_base_data()
    
    print("\n" + "="*50)
    print("PRODUCTION RUN: BEST FLAT MODEL")
    print(f"Fast Mode: {FAST_MODE}")
    print("="*50)
    
    indices = get_feature_indices(FINAL_FEATS, FINAL_MONTHS)
    X = get_feature_matrix(num_samples, indices)
    
    configs = [
        ("XGB_512L_100E_10P_AllYear (Power 0.5, Quad Scale, Spatial Aug, 100 Est, 10 Parallel)",
         {"_balance_power": 0.5, "_scale_data": 4, "_spatial_aug": True,
          "max_leaves": 512, "n_estimators": 100, "num_parallel_tree": 10,
          "subsample": 0.8, "colsample_bynode": 0.8}),
    ]
    
    results_list = []
    y_species = [row['l3'] for row in y_df]
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    for name, extra in configs:
        print(f"\nRunning Production Experiment: {name}")
        
        fold_args = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_species)):
            extra_copy = extra.copy()
            fold_args.append((fold, train_idx, val_idx, X, y_df, AVAILABLE_GPUS[fold % len(AVAILABLE_GPUS)], extra_copy, name))
        
        start_t = time.time()
        with mp.Pool(processes=NUM_FOLDS) as pool:
            all_res = pool.map(run_single_fold, fold_args)
        elapsed = time.time() - start_t
        
        avg_res = {k: np.mean([r[k] for r in all_res]) for k in all_res[0].keys()}
        avg_res['name'] = name
        results_list.append(avg_res)
        print(f"L3 Acc: {avg_res['l3_acc']:.4f}, L3 Bal Acc: {avg_res['l3_bal_acc']:.4f} (Took {elapsed:.1f}s)")

    print("\n" + "="*80)
    print(f"{'L3 ACC':<10} | {'L3 BAL ACC':<12} | {'MODEL NAME'}")
    print("-" * 80)
    for r in results_list:
        print(f"{r['l3_acc']:.4f}     | {r['l3_bal_acc']:.4f}     | {r['name']}")
    print("="*80)#!/usr/bin/env python3
"""
TreeSatAI - Flat Hierarchical Tree Species Classification with XGBoost

This script implements a standard flattened multiclass approach using the 
same "ultimate" feature set and parameters as the Top-Down model for comparison.
"""

import argparse
import logging
import numpy as np
import netCDF4 as nc
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import time
import warnings
import multiprocessing as mp
from collections import Counter
def run_single_fold(args):
    fold, train_idx, val_idx, X, y_df, gpu_id, extra_params, name, disable_scaling = args
logger = logging.getLogger(__name__)
    logger.info("\n--- FOLD %s Starting on GPU %s ---", fold+1, gpu_id)
# =============================================
# CONFIGURATION
# =============================================
DATA_DIR = Path('data/')
    X_train, y_train_struct = apply_augmentation(X_train, y_train_struct, extra_params, logger)
    X_train, X_val, scaler = scale_data(X_train, X_val, disable_scaling)
    y_df = []
    for label in labels_raw:
        if isinstance(label, bytes): label = label.decode('utf-8')
        path = species_path_map.get(label, species_path_map['european beech'])
        y_df.append(path)
    return num_samples, y_df

def get_feature_matrix(num_samples, selected_indices, s2_nc: Path, dem_nc: Path):
    with nc.Dataset(s2_nc, 'r') as ds_s2, nc.Dataset(dem_nc, 'r') as ds_dem:
        s2_var = 'X' if 'X' in ds_s2.variables else 'feature'
        X_s2 = ds_s2.variables[s2_var][:num_samples, :, :, selected_indices]
        X_dem = ds_dem.variables['patches'][:num_samples, :, :, :]
    X_s2 = np.transpose(X_s2, (0, 3, 1, 2)).reshape(num_samples, -1)
    X_dem = np.transpose(X_dem, (0, 3, 1, 2)).reshape(num_samples, -1)
    X = np.concatenate([X_s2, X_dem], axis=1)
    X[X < -500] = 0
    return X


def parse_months(months_arg: str):
    months = set()
    for part in months_arg.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            months.update(range(int(start), int(end) + 1))
        else:
            months.add(int(part))
    valid = sorted(m for m in months if 1 <= m <= 12)
    if not valid:
        raise ValueError(f"No valid months parsed from {months_arg}")
    return valid


def load_dataset(s2_nc: Path, dem_nc: Path, selected_indices):
    num_samples, y_df = load_base_data(s2_nc, dem_nc)
    X = get_feature_matrix(num_samples, selected_indices, s2_nc, dem_nc)
    return X, y_df


def apply_augmentation(X_train, y_train_struct, extra_params, logger=None):
    if not AUGMENT:
        return X_train, y_train_struct

    scale_factor = extra_params.pop('_scale_data', 1)
    aug_all = extra_params.pop('_aug_all', False)
    spatial_aug = extra_params.pop('_spatial_aug', False)
    balance_power = extra_params.pop('_balance_power', 1.0)

    if scale_factor > 1 or aug_all or balance_power > 0 or spatial_aug:
        if logger:
            logger.debug("Applying custom augmentation settings for scale=%s aug_all=%s balance_power=%s spatial=%s",
                         scale_factor, aug_all, balance_power, spatial_aug)
        l3_labels = [row['l3'] for row in y_train_struct]
        counts = Counter(l3_labels)
        max_count = max(counts.values())
        X_extra, y_extra = [], []

        for label, count in counts.items():
            target_count = int(count * (max_count / count) ** balance_power)
            if target_count > count:
                num_to_add = target_count - count
                idx = [i for i, l in enumerate(l3_labels) if l == label]
                extra_idx = np.random.choice(idx, num_to_add, replace=True)
                X_extra.append(X_train[extra_idx])
                y_extra.extend([y_train_struct[i] for i in extra_idx])

        if X_extra:
            X_train = np.concatenate([X_train] + X_extra, axis=0)
            y_train_struct = y_train_struct + y_extra

        if scale_factor > 1:
            X_orig = X_train.copy()
            y_orig = y_train_struct.copy()
            for _ in range(scale_factor - 1):
                X_train = np.concatenate([X_train, X_orig], axis=0)
                y_train_struct = y_train_struct + y_orig

        if spatial_aug:
            num_c = X_train.shape[1] // 9
            X_temp = X_train.reshape(-1, num_c, 3, 3)

            rots = np.random.randint(0, 4, size=len(X_temp))
            for k in range(1, 4):
                idx = np.where(rots == k)[0]
                if len(idx) > 0:
                    X_temp[idx] = np.rot90(X_temp[idx], k=k, axes=(2, 3))

            flip_h = np.random.random(len(X_temp)) > 0.5
            idx_h = np.where(flip_h)[0]
            if len(idx_h) > 0:
                X_temp[idx_h] = np.flip(X_temp[idx_h], axis=3)

            flip_v = np.random.random(len(X_temp)) > 0.5
            idx_v = np.where(flip_v)[0]
            if len(idx_v) > 0:
                X_temp[idx_v] = np.flip(X_temp[idx_v], axis=2)

            X_train = X_temp.reshape(len(X_temp), -1)

        if aug_all:
            X_train = X_train + np.random.normal(0, NOISE_STD, X_train.shape)
    else:
        X_train, y_train_struct = oversample_and_augment(X_train, y_train_struct)

    return X_train, y_train_struct


def scale_data(X_train, X_val, disable_scaling):
    if disable_scaling:
        logger.debug("Skipping StandardScaler (disabled by request)")
        return X_train, X_val, None
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler


def evaluate_predictions(y_struct, p3_pred, species_path_map):
    g1 = [row['l1'] for row in y_struct]
    g2 = [row['l2'] for row in y_struct]
    g3 = [row['l3'] for row in y_struct]
    p1 = [species_path_map[p]['l1'] for p in p3_pred]
    p2 = [species_path_map[p]['l2'] for p in p3_pred]
    return {
        'l1_acc': accuracy_score(g1, p1),
        'l2_acc': accuracy_score(g2, p2),
        'l3_acc': accuracy_score(g3, p3_pred),
        'l3_bal_acc': balanced_accuracy_score(g3, p3_pred)
    }

def oversample_and_augment(X_in, y_in):
    l3_labels = [row['l3'] for row in y_in]
    counts = Counter(l3_labels)
    max_count = max(counts.values())
    X_extra, y_extra = [], []
    for label, count in counts.items():
        if count < max_count:
            num_to_add = max_count - count
            class_indices = [i for i, l in enumerate(l3_labels) if l == label]
            extra_indices = np.random.choice(class_indices, num_to_add, replace=True)
            X_sub = X_in[extra_indices].copy()
            y_sub = [y_in[i] for i in extra_indices]
            X_sub = X_sub + np.random.normal(0, NOISE_STD, X_sub.shape)
            X_extra.append(X_sub)
            y_extra.extend(y_sub)
    if X_extra:
        return np.concatenate([X_in] + X_extra, axis=0), y_in + y_extra
    return X_in, y_in

def run_single_fold(args):
    fold, train_idx, val_idx, X, y_df, gpu_id, extra_params, name = args
    print(f"\n--- FOLD {fold+1} Starting on GPU {gpu_id} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_struct = [y_df[i] for i in train_idx]
    y_val_struct = [y_df[i] for i in val_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Extract internal flags
    scale_factor = extra_params.pop('_scale_data', 1)
    aug_all = extra_params.pop('_aug_all', False)
    spatial_aug = extra_params.pop('_spatial_aug', False)
    balance_power = extra_params.pop('_balance_power', 1.0) # 1.0 = Full, 0.0 = None
    
    if AUGMENT:
        if scale_factor > 1 or aug_all or balance_power > 0 or spatial_aug:
             # Custom logic for scaling/aug_all
             l3_labels = [row['l3'] for row in y_train_struct]
             counts = Counter(l3_labels)
             max_count = max(counts.values())
             X_extra, y_extra = [], []
             
             for label, count in counts.items():
                 # target = count * (max/count)^power
                 target_count = int(count * (max_count / count) ** balance_power)
                 if target_count > count:
                     num_to_add = target_count - count
                     idx = [i for i, l in enumerate(l3_labels) if l == label]
                     extra_idx = np.random.choice(idx, num_to_add, replace=True)
                     X_extra.append(X_train[extra_idx])
                     y_extra.extend([y_train_struct[i] for i in extra_idx])
             
             if X_extra:
                 X_train = np.concatenate([X_train] + X_extra, axis=0)
                 y_train_struct = y_train_struct + y_extra
             
             if scale_factor > 1:
                 X_orig = X_train.copy()
                 y_orig = y_train_struct.copy()
                 for _ in range(scale_factor - 1):
                     X_train = np.concatenate([X_train, X_orig], axis=0)
                     y_train_struct = y_train_struct + y_orig
             
             if spatial_aug:
                 # Reshape to (N, C, 3, 3) to apply spatial transforms
                 num_c = X_train.shape[1] // 9
                 X_temp = X_train.reshape(-1, num_c, 3, 3)
                 
                 # Vectorized Rotation/Flips (MUCH faster than looping)
                 rots = np.random.randint(0, 4, size=len(X_temp))
                 for k in range(1, 4):
                     idx = np.where(rots == k)[0]
                     if len(idx) > 0:
                         X_temp[idx] = np.rot90(X_temp[idx], k=k, axes=(2, 3))
                 
                 # Random flips
                 flip_h = np.random.random(len(X_temp)) > 0.5
                 idx_h = np.where(flip_h)[0]
                 if len(idx_h) > 0:
                     X_temp[idx_h] = np.flip(X_temp[idx_h], axis=3) # Horizontal
                     
                 flip_v = np.random.random(len(X_temp)) > 0.5
                 idx_v = np.where(flip_v)[0]
                 if len(idx_v) > 0:
                     X_temp[idx_v] = np.flip(X_temp[idx_v], axis=2) # Vertical
                         
                 X_train = X_temp.reshape(len(X_temp), -1)

             if aug_all:
                 X_train = X_train + np.random.normal(0, NOISE_STD, X_train.shape)
        else:
             # Standard balanced oversampling
             X_train, y_train_struct = oversample_and_augment(X_train, y_train_struct)
    
    # Flat Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform([row['l3'] for row in y_train_struct])
    y_val = le.transform([row['l3'] for row in y_val_struct])
    g3 = [row['l3'] for row in y_val_struct] # Ground truth species names
    
    is_prod = not FAST_MODE
    xgb_defaults = {
        'n_estimators': 100,
        'max_depth': 6 if FAST_MODE else 0, # Rely on max_leaves for lossguide
        'max_leaves': 1023, # High complexity for production
        'grow_policy': 'lossguide' if is_prod else 'depthwise', 
        'learning_rate': 0.1,
        'tree_method': 'hist',
        'device': f'cuda:{gpu_id}',
        'random_state': SEED,
        'n_jobs': 16, # Scaling up CPU usage for faster data prep
        'early_stopping_rounds': 50 if is_prod else None
    }
    xgb_defaults.update(extra_params)
    
    clf = xgb.XGBClassifier(**xgb_defaults)
    
    if is_prod:
        # Enable maximum verbosity to see every single round
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1)
    else:
        clf.fit(X_train, y_train)
    
    # Save Model Checkpoint
    safe_name = name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
    checkpoint_path = CHECKPOINT_DIR / f"{safe_name}_fold{fold+1}.json"
    clf.save_model(str(checkpoint_path))
    
    p3_idx = clf.predict(X_val)
    p3 = le.inverse_transform(p3_idx)
    
    # Map predictions back to hierarchy for parity metrics
    species_path_map = get_mappings()
    p1 = [species_path_map[p]['l1'] for p in p3]
    p2 = [species_path_map[p]['l2'] for p in p3]
    g1 = [row['l1'] for row in y_val_struct]
    g2 = [row['l2'] for row in y_val_struct]
    
    res = {
        'l1_acc': accuracy_score(g1, p1),
        'l2_acc': accuracy_score(g2, p2),
        'l3_acc': accuracy_score(g3, p3),
        'l3_bal_acc': balanced_accuracy_score(g3, p3)
    }
    logger.info("Fold %s Finished: L3 Acc: %.4f, L3 Bal Acc: %.4f", fold+1, res['l3_acc'], res['l3_bal_acc'])
    return res

if __name__ == "__main__":
    num_samples, y_df = load_base_data()
    
    print("\n" + "="*50)
    print("PRODUCTION RUN: BEST FLAT MODEL")
    print(f"Fast Mode: {FAST_MODE}")
    print("="*50)
    
    indices = get_feature_indices(FINAL_FEATS, FINAL_MONTHS)
    X = get_feature_matrix(num_samples, indices)
    
    # FINAL ULTIMATE CHAMPION: 512 Leaves, 100 Estimators, 10 Parallel Trees
    configs = [
        ("XGB_512L_100E_10P_AllYear (Power 0.5, Quad Scale, Spatial Aug, 100 Est, 10 Parallel)", 
         {"_balance_power": 0.5, "_scale_data": 4, "_spatial_aug": True, 
          "max_leaves": 512, "n_estimators": 100, "num_parallel_tree": 10,
          "subsample": 0.8, "colsample_bynode": 0.8}),
    ]
    
    results_list = []
    y_species = [row['l3'] for row in y_df]
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    for name, extra in configs:
        print(f"\nRunning Production Experiment: {name}")
        
        fold_args = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_species)):
            # Force deep copy of extra to prevent pop() from affecting next folds
            extra_copy = extra.copy()
            fold_args.append((fold, train_idx, val_idx, X, y_df, AVAILABLE_GPUS[fold % len(AVAILABLE_GPUS)], extra_copy, name))
        
        start_t = time.time()
        with mp.Pool(processes=NUM_FOLDS) as pool:
            all_res = pool.map(run_single_fold, fold_args)
        elapsed = time.time() - start_t
        
        avg_res = {k: np.mean([r[k] for r in all_res]) for k in all_res[0].keys()}
        avg_res['name'] = name
        results_list.append(avg_res)
        print(f"L3 Acc: {avg_res['l3_acc']:.4f}, L3 Bal Acc: {avg_res['l3_bal_acc']:.4f} (Took {elapsed:.1f}s)")

    print("\n" + "="*80)
    print(f"{'L3 ACC':<10} | {'L3 BAL ACC':<12} | {'MODEL NAME'}")
    print("-" * 80)
    for r in results_list:
        print(f"{r['l3_acc']:.4f}     | {r['l3_bal_acc']:.4f}     | {r['name']}")
    print("="*80)
