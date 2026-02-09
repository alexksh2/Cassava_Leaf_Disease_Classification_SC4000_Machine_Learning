# SC4000 Machine Learning: Cassava Leaf Disease Classification

Multi-class image classification system for identifying cassava leaf diseases using a two-layer stacking ensemble of deep convolutional neural networks and gradient-boosted decision trees.

## Problem Formulation

**Task:** 5-class classification of cassava leaf pathologies from RGB imagery.

| Class | Label |
|-------|-------|
| 0 | Cassava Bacterial Blight (CBB) |
| 1 | Cassava Brown Streak Disease (CBSD) |
| 2 | Cassava Green Mottle (CGM) |
| 3 | Cassava Mosaic Disease (CMD) |
| 4 | Healthy |

**Dataset:** ~27,053 labelled JPEG images with inherent class imbalance. Stratified 80/10/10 train/validation/test split. Both balanced (undersampled) and imbalanced variants are maintained for ablation.

## Architecture

### Layer 1 — Base Learners

Eight heterogeneous deep learning models trained independently as feature extractors and soft-vote classifiers:

| Model | Input Resolution | Framework | Backbone Source |
|-------|-----------------|-----------|-----------------|
| Custom CNN | 128x128 | PyTorch | — |
| AlexNet | 128x128 | PyTorch | torchvision |
| ResNet-50 | 128x128 | PyTorch | torchvision |
| ResNeXt-50 (32x4d) | 128x128 | PyTorch | timm |
| EfficientNet-B0 | 128x128 | PyTorch | torchvision |
| EfficientNet-B4 | 299x299 | PyTorch | torchvision |
| Inception-V3 | 299x299 | PyTorch | torchvision |
| Vision Transformer (ViT) | 224x224 | PyTorch / HuggingFace | timm |

Additionally, TensorFlow-based CropNet and EfficientNet models are used via inference-only pipelines.

**Training configuration:**
- Optimizer: SGD (momentum=0.9) / Adam, lr=1e-3
- Scheduler: `ReduceLROnPlateau`
- Loss: `CrossEntropyLoss`
- Epochs: 20–30, batch size: 128
- Per-channel normalization (dataset-computed mean/std)

Each base learner outputs a 5-dimensional softmax probability vector over the validation and test partitions, serving as the input feature space for Layer 2.

### Layer 2 — Meta-Learners (Stacking Ensemble)

Gradient-boosted tree models trained on the concatenated probability outputs of the top-performing Layer 1 models (ResNeXt, ViT, EfficientNet, CropNet):

| Meta-Learner | Library | Key Hyperparameters |
|--------------|---------|---------------------|
| LightGBM | lightgbm | 1000 rounds, lr=0.01, num_leaves=10, early stopping (patience=10) |
| XGBoost | xgboost | Gradient boosting, `multi:softprob` objective |
| CatBoost | catboost | 1000 iterations, lr=0.01, logloss |
| Logistic Regression | scikit-learn | Baseline linear meta-learner |

**Layer 2 extension:** A secondary stacking stage combines LightGBM ensemble predictions with logistic regression outputs via 5-fold cross-validated LightGBM, yielding the final prediction.

## Pipeline

```
Raw Images
    │
    ▼
[Data Preparation] ── stratified split ── train / val / test CSVs
    │
    ▼
[Normalization] ── per-channel mean & std (computed on train set)
    │
    ▼
[Layer 1 Training] ── 8 independent CNN / ViT models
    │
    ▼
[Layer 1 Inference] ── 5-class probability vectors on val & test sets
    │
    ▼
[Layer 2 Stacking] ── LightGBM / XGBoost / CatBoost on concatenated probabilities
    │
    ▼
[Layer 2 Extension] ── 5-fold CV stacking of L1 ensemble outputs
    │
    ▼
[Evaluation] ── Accuracy, Precision, Recall, F1, Log Loss, Confusion Matrix
```

## Project Structure

```
.
├── data/                       # Train/val/test CSVs (balanced & imbalanced)
├── data-preprocess/            # EDA, stratified splitting, undersampling notebooks
│   ├── Cassava_EDA.ipynb
│   ├── data_prep.ipynb
│   ├── data_split.ipynb
│   ├── data_undersample.ipynb
│   └── data_count.ipynb
├── scripts/                    # Training, inference, and ensemble entry points
│   ├── CNN.py, AlexNet.py      # Custom / classic architectures
│   ├── Resnet.py, ResNext_v2.py
│   ├── EfficientnetB0.py, EfficientnetB4.py
│   ├── Inception.py
│   ├── vision_transformer.py
│   ├── *_inference.py          # Per-model inference pipelines
│   ├── ensemble_lgbm.py        # LightGBM stacking (L1)
│   ├── ensemble_lgbm_l2.py     # LightGBM stacking (L2)
│   ├── ensemble_xgboost.py     # XGBoost stacking
│   ├── ensemble_catboost.py    # CatBoost stacking
│   └── ensemble_catboost_cv.py # CatBoost with cross-validation
├── notebook/                   # Exploratory and experimental notebooks
├── utils/
│   ├── generate_dataset.py     # Image aggregation & label mapping
│   ├── calc_mean_std.py        # Per-channel normalization statistics
│   └── train_test_split.py     # Stratified partitioning
├── output/                     # Timestamped experiment artifacts
│   ├── <model>_<timestamp>/    # Weights (.pth/.keras), logs, prediction CSVs
│   ├── ensemble_lightgbm/
│   ├── ensemble_lightgbm_l2/
│   ├── ensemble_xgboost/
│   └── ensemble_catboost/
└── archive/                    # Deprecated experiments
```

## Evaluation Metrics

- **Accuracy** — overall classification correctness
- **Precision / Recall / F1-score** — per-class and macro-averaged
- **F-beta score** — weighted harmonic mean for class-imbalanced evaluation
- **Log loss** — calibration quality of predicted probability distributions
- **Confusion matrix** — per-class error distribution analysis

## Dependencies

| Category | Libraries |
|----------|-----------|
| Deep Learning | PyTorch, torchvision, timm, TensorFlow/Keras, HuggingFace Transformers |
| Gradient Boosting | LightGBM, XGBoost, CatBoost |
| ML Utilities | scikit-learn |
| Image Processing | OpenCV, Pillow |
| Data & Viz | pandas, NumPy, matplotlib |

## Reproducibility

Each experiment run persists to a timestamped directory under `output/` containing:
- `best_model.pth` / `.keras` — serialized model weights at best validation loss
- `training.log` — epoch-level loss and metric traces
- Validation/test prediction CSVs with per-class probabilities
- Confusion matrix visualizations
