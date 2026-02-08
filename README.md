# Tree Classifiers - Satellite-Based Tree Species Classification

A machine learning project for classifying tree species using Sentinel-2 satellite imagery and deep Earth observation data.

## Project Overview

This repository contains Python scripts for hierarchical tree species classification using satellite data. The project uses Sentinel-2 satellite imagery combined with vegetation indices and GEE (Google Earth Engine) data to classify trees at three levels:

- **L1 (Leaf Type)**: Broadleaf vs Needleleaf
- **L2 (Genus)**: Beech, Oak, Pine, etc.
- **L3 (Species)**: European Beech, Scots Pine, etc.

## Scripts

### 1. `explore_tree_data.py`
Data exploration and preparation script for CNN classification.

**Features:**
- Explores GeoJSON data structure
- Analyzes tree species distribution
- Handles Sentinel-2 satellite features (10 bands + 5 vegetation indices)
- Processes temporal data (8 monthly composites)
- Manages spatial patches (5×5 resolution)
- Prepares data for CNN input format

**Output:**
Processed datasets ready for neural network training

### 2. `train_tree_classifier.py`
Hierarchical tree species classification using Convolutional Neural Networks (CNN).

**Features:**
- Implements hierarchical classification (L1 → L2 → L3)
- Class-weighted loss function for imbalanced data
- Data augmentation (flips, rotations)
- K-fold cross-validation
- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization

**Key Parameters:**
- Batch Size: 64
- Epochs: 100
- Learning Rate: 0.001
- CV Folds: 5

### 3. `train_tree_classifier_xgb_flat.py`
Flat multiclass tree species classification using XGBoost.

**Features:**
- Standard flattened multiclass approach
- XGBoost gradient boosting algorithm
- Data augmentation with noise injection
- Stratified K-fold cross-validation
- GPU acceleration support
- Checkpoint saving for model persistence

**Key Parameters:**
- CV Folds: 5
- Data Augmentation: Enabled
- Noise Std: 0.01

## Data Requirements

The scripts expect Sentinel-2 satellite data with the following structure:

- **Spectral Bands**: B2-B12, B8A (10 bands)
- **Vegetation Indices**: NDVI, EVI, EVI2, NDWI, SAVI
- **Temporal Dimension**: 8 monthly composites (March-October)
- **Spatial Dimension**: 5×5 patches (2-meter resolution)
- **Format**: GeoJSON or NetCDF files

## Installation

### Requirements
- Python 3.7+
- PyTorch (for CNN model)
- XGBoost (for XGBoost model)
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

### Setup
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn xgboost netCDF4
```

## Usage

### Run Data Exploration
```bash
python explore_tree_data.py
```

### Train CNN Classifier
```bash
python train_tree_classifier.py
```

### Train XGBoost Classifier
```bash
python train_tree_classifier_xgb_flat.py
```

## Output

The scripts generate:
- Training/validation metrics and reports
- Confusion matrices for classification evaluation
- Model checkpoints for best performance
- Detailed classification reports per fold

## Implementation Notes

- **GPU Support**: Uses CUDA when available, falls back to CPU
- **Class Imbalance**: Handles via class weights and balanced accuracy metrics
- **Cross-Validation**: Stratified K-fold to maintain class distribution
- **Data Augmentation**: Random flips, rotations, and noise injection

## Author

Mays Alsheikh - Third Semester Project on Tree Classification with Deep Earth Observation

## License

Open source - feel free to use and modify

---

**Project Focus**: Combining satellite imagery with machine learning for automated tree species identification in Earth observation applications.
