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

Ten deep learning models were trained and evaluated independently. After performance comparison, **4 were selected** for the stacking ensemble; the remaining 6 were discarded.

**Selected for ensemble (predictions feed into Layer 2):**

| Model | Input Resolution | Framework | Backbone Source | Used In Ensemble |
|-------|-----------------|-----------|-----------------|:---:|
| ResNeXt-50 (32x4d) | 128x128 | PyTorch | timm | Yes |
| Vision Transformer (ViT) | 224x224 | PyTorch / HuggingFace | timm | Yes |
| EfficientNet (TF) | 224x224 | TensorFlow | TF Hub | Yes |
| CropNet (TF) | 224x224 | TensorFlow | TF Hub | Yes |

**Evaluated but not selected:**

| Model | Input Resolution | Framework | Backbone Source | Used In Ensemble |
|-------|-----------------|-----------|-----------------|:---:|
| Custom CNN | 128x128 | PyTorch | — | No |
| AlexNet | 128x128 | PyTorch | torchvision | No |
| ResNet-50 | 128x128 | PyTorch | torchvision | No |
| EfficientNet-B0 | 128x128 | PyTorch | torchvision | No |
| EfficientNet-B4 | 299x299 | PyTorch | torchvision | No |
| Inception-V3 | 299x299 | PyTorch | torchvision | No |

**Training configuration (all models):**
- Optimizer: SGD (momentum=0.9) / Adam, lr=1e-3
- Scheduler: `ReduceLROnPlateau`
- Loss: `CrossEntropyLoss`
- Epochs: 20–30, batch size: 128
- Per-channel normalization (dataset-computed mean/std)

Each selected model outputs a 5-dimensional softmax probability vector over the validation and test partitions, yielding a **20-feature** input matrix (4 models x 5 classes) for the Layer 2 meta-learners.

### Layer 2 — Meta-Learners (Stacking Ensemble)

Gradient-boosted tree models trained on the concatenated softmax outputs of the 4 selected Layer 1 models:

| Meta-Learner | Library | Key Hyperparameters | Role |
|--------------|---------|---------------------|------|
| LightGBM | lightgbm | 1000 rounds, lr=0.01, num_leaves=10, early stopping (patience=10) | L1 ensemble (output feeds L2) |
| XGBoost | xgboost | Gradient boosting, `multi:softprob` objective, lr=0.01 | L1 ensemble (alternative) |
| CatBoost | catboost | 1000 iterations, lr=0.01, logloss | L1 ensemble (alternative) |
| Logistic Regression | scikit-learn | Baseline linear meta-learner | L1 ensemble (output feeds L2) |

### Layer 2 Extension — Second-Stage Stacking

A secondary LightGBM meta-learner is trained on the **concatenated probability outputs of the L1 LightGBM and L1 Logistic Regression** ensembles (10 features = 2 ensembles x 5 classes), using 5-fold stratified cross-validation with early stopping. This produces the final prediction.

## Pipeline

```
Raw Images
    │
    ▼
[Data Preparation] ── stratified 80/10/10 split ── train / val / test CSVs
    │
    ▼
[Normalization] ── per-channel mean & std (computed on train set)
    │
    ▼
[Layer 1 Training] ── 10 independent CNN / ViT models
    │
    ▼
[Model Selection] ── 4 selected: ResNeXt, ViT, EfficientNet (TF), CropNet (TF)
    │                  6 discarded: CNN, AlexNet, ResNet, EfficientNet-B0/B4, Inception
    ▼
[Layer 1 Inference] ── 5-class softmax vectors on val & test sets (20 features total)
    │
    ▼
[L1 Stacking] ── LightGBM / XGBoost / CatBoost / LogReg on concatenated probabilities
    │
    ├── LightGBM L1 output (5 probs) ──┐
    └── LogReg L1 output (5 probs) ────┤
                                        ▼
[L2 Stacking] ── LightGBM on 10 features (LGBM + LogReg), 5-fold CV
    │
    ▼
[Evaluation] ── Accuracy, Precision, Recall, F1, F-beta, Log Loss, Confusion Matrix
```

## Project Structure

```
.
├── data/                           # Train/val/test CSVs (balanced & imbalanced)
├── data-preprocess/                # EDA, stratified splitting, undersampling notebooks
│   ├── Cassava_EDA.ipynb
│   ├── data_prep.ipynb
│   ├── data_split.ipynb
│   ├── data_undersample.ipynb
│   └── data_count.ipynb
├── scripts/
│   ├── ensemble_pipeline/          # Scripts used in the two-layer stacking ensemble
│   │   ├── ResNext_v2.py           # ResNeXt-50 training
│   │   ├── vision_transformer.py   # ViT training
│   │   ├── resnext_inference.py    # ResNeXt inference → L1 probabilities
│   │   ├── vit_inference.py        # ViT inference → L1 probabilities
│   │   ├── efficientnet_inference.py # EfficientNet (TF) inference → L1 probabilities
│   │   ├── cropnet_inference.py    # CropNet (TF) inference → L1 probabilities
│   │   ├── ensemble_lgbm.py       # L1 LightGBM meta-learner
│   │   ├── ensemble_lgbm_kfold.py # L1 LightGBM with k-fold CV
│   │   ├── ensemble_xgboost.py    # L1 XGBoost meta-learner
│   │   ├── ensemble_catboost.py   # L1 CatBoost meta-learner
│   │   ├── ensemble_catboost_cv.py # L1 CatBoost with cross-validation
│   │   ├── ensemble_average_logreg.ipynb # L1 Logistic Regression baseline
│   │   └── ensemble_lgbm_l2.py    # L2 LightGBM (stacks L1 LGBM + LogReg outputs)
│   └── unused/                     # Models not selected for the final ensemble
│       ├── AlexNet.py
│       ├── CNN.py
│       ├── Resnet.py
│       ├── EfficientnetB0.py
│       ├── EfficientnetB4.py
│       ├── EfficientnetB4_v2.py
│       ├── EfficientnetB4_final.py
│       ├── Inception.py
│       └── vision_transformer_v2.py
├── notebook/                       # Exploratory and experimental notebooks
├── utils/
│   ├── generate_dataset.py         # Image aggregation & label mapping
│   ├── calc_mean_std.py            # Per-channel normalization statistics
│   └── train_test_split.py         # Stratified partitioning
├── output/                         # Timestamped experiment artifacts
│   ├── <model>_<timestamp>/        # Weights (.pth/.keras), logs, prediction CSVs
│   ├── ensemble_lightgbm/
│   ├── ensemble_lightgbm_l2/
│   ├── ensemble_xgboost/
│   └── ensemble_catboost/
└── archive/                        # Deprecated experiments
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
