# RVL-CDIP Document Classification System

## Overview
This project implements a transformer-based document classification system using the RVL-CDIP dataset, designed to accurately categorize financial and business documents such as invoices, forms, budgets, and memos. Leveraging pre-trained Vision Transformer (ViT) and Swin Transformer (Swin-Tiny) models, fine-tuned on progressively larger dataset subsets, the system achieves robust performance within Google Colab’s free-tier constraints. Experiments with ViT-Base and Swin-Tiny provide a comparative analysis of global versus hierarchical processing for document classification.

## Task Requirements
- **Objective**: Develop a transformer-based system to classify RVL-CDIP document images into 16 categories, emphasizing financial document types (invoices, receipts, tax forms, financial reports).
- **Dataset**: `aharley/rvl_cdip` from Hugging Face (400,000 grayscale images, 16 classes: 320,000 train, 40,000 validation, 40,000 test).
- **Financial Focus**: Prioritizes categories like invoices (label 11), forms (label 1, proxy for tax forms), budgets (label 10, proxy for financial reports), and memos (label 15, proxy for financial reports).

## Model Selection
- **Vision Transformer (ViT-Base)**:
  - **Model**: `google/vit-base-patch16-224-in21k`.
  - **Rationale**: Pre-trained on ImageNet-21k, ViT captures global image context effectively, ideal for document classification where overall structure matters. Its transformer architecture meets the assignment’s core requirement.
- **Swin-Tiny**:
  - **Model**: `microsoft/swin-tiny-patch4-window7-224`.
  - **Rationale**: Swin’s hierarchical processing, with shifted window attention, excels at detecting structured layouts (e.g., tables in invoices), offering a comparative test against ViT’s global approach.

## Methodology
- **Dataset Preparation**:
  - **Training**: Subsets of 5,000, 10,000, and 20,000 samples streamed from 320,000 training images.
  - **Validation**: 2,000 samples (~125 per class) streamed from 40,000 validation images.
  - **Test**: 2,000 samples (~125 per class) streamed from 40,000 test images.
  - Images converted to RGB and resized (max 224px) to match model input requirements.
- **Data Balancing**:
  - **Initial Run (5,000 Samples)**: Random sampling resulted in uneven class distribution, limiting performance to 59.15% accuracy.
  - **Post-5,000 Improvement**: Introduced balanced sampling via a custom `BalancedStreamingDataset` class:
    - **Target**: Equal representation (~312 samples/class for 5,000, ~625 for 10,000, ~1,250 for 20,000).
    - **Method**: Iterated through the dataset, yielding samples only until each class reached its target, ensuring fairness across all 16 categories.
    - **Impact**: Improved accuracy (e.g., 77.2% at 10,000) and financial class performance by mitigating bias toward dominant classes.
- **Preprocessing**:
  - Applied augmentation: random rotation (10°), affine translation (10%), and color jitter (brightness/contrast 0.2) to enhance generalization across varied document appearances.
- **Fine-Tuning**:
  - **Optimizer**: AdamW with weight decay of 0.01 for regularization.
  - **Batch Size**: Effective batch size of 64 via gradient accumulation (per-device batch size 8, 8 steps) to fit VRAM limits.
  - **Epochs**: Up to 7, with early stopping (patience=2) based on validation accuracy to prevent overfitting.
  - **Mixed Precision**: FP16 enabled for faster computation and lower memory usage.
- **Hyperparameter Tuning**:
  - **Learning Rate**: Initially 5e-5 (standard for ViT), reduced to 3e-5 after 5,000-sample run to stabilize training with larger datasets, balancing convergence and accuracy.
  - **Batch Size**: Increased from 32 (5,000 samples) to 64 (10,000+ samples) via gradient accumulation, improving gradient stability within VRAM constraints.
  - **Early Stopping**: Added post-5,000 run (patience=2), stopping when validation accuracy plateaued (e.g., epoch 6 for 20,000 samples).
  - **Outcome**: Tuning lifted accuracy from 59.15% (5,000) to 78.85% (ViT) and 79.55% (Swin), with consistent gains in financial class performance.
- **Evaluation**: Computed accuracy, precision, recall, and F1 on 2,000-sample validation and test sets, with per-class accuracy for financial categories.

## Constraints
- **Environment**: Google Colab free tier with T4 GPU.
  - **VRAM**: 15 GB (peak usage: ~5-6 GB during training, ~1 GB for ViT and ~335 MB for Swin post-training).
  - **RAM**: ~12 GB (streamed data to avoid loading full dataset).
  - **Disk**: ~20-24 GB (ViT model: 328 MB, Swin model: 106 MB, checkpoints: ~1-2 GB).
- **Runtime**: Limited to ~12 hours; ViT 20,000-sample run took ~3 hours 7 minutes, Swin runtime similar (exact time not provided but within bounds).
- **Solutions**:
  - Streamed dataset to manage RAM limits.
  - Used FP16 and gradient accumulation to optimize VRAM usage.
  - Saved models/checkpoints efficiently to fit disk space.

## Results
### ViT-Base Experiments
- **5,000 Samples**: Initial run achieved 59.15% accuracy (unbalanced data).
- **10,000 Samples**: Improved to 77.2% with balanced sampling and early stopping.
- **20,000 Samples**:
  - **Training**: 20,000 training samples, 2,000 validation samples.
    ```
    Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1
    1     | 1.4753        | 1.4567          | 0.6325   | 0.6540    | 0.6325 | 0.6218
    2     | 1.1496        | 1.1631          | 0.7005   | 0.7139    | 0.7005 | 0.6991
    3     | 0.9907        | 1.0470          | 0.7205   | 0.7474    | 0.7205 | 0.7234
    4     | 0.8442        | 0.9206          | 0.7575   | 0.7790    | 0.7575 | 0.7615
    5     | 0.7979        | 0.8558          | 0.7770   | 0.7918    | 0.7770 | 0.7801
    6     | 0.6795        | 0.8019          | 0.7895   | 0.7948    | 0.7895 | 0.7907
    ```
    - Training Time: ~3 hours 7 minutes.
    - GPU Memory Post-Training: 1000 MB (~1 GB).
  - **Test Evaluation**: 2,000 test samples.
    ```
    Test Results:
    - Eval Loss: 0.8202
    - Accuracy: 0.7885 (78.85%)
    - Precision: 0.7906
    - Recall: 0.7885
    - F1: 0.7888
    - Runtime: ~451 seconds (7.5 minutes)
    ```
    - **Financial Class Accuracy**:
      - Form (label 1, tax forms proxy): 61.60%.
      - Budget (label 10, financial reports proxy): 62.40%.
      - Invoice (label 11): 72.80%.
      - Memo (label 15, financial reports proxy): 71.20%.
  - **Model Size**: 328 MB on disk.

### Swin-Tiny Experiment
- **20,000 Samples**: 20,000 training samples, 2,000 validation samples, 2,000 test samples.
  - **Training**:
    ```
    Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1
    1     | 1.1350        | 1.1454          | 0.6570   | 0.6927    | 0.6570 | 0.6566
    2     | 0.8994        | 0.9141          | 0.7245   | 0.7398    | 0.7245 | 0.7237
    3     | 0.7527        | 0.8515          | 0.7450   | 0.7607    | 0.7450 | 0.7460
    4     | 0.6534        | 0.7543          | 0.7790   | 0.7874    | 0.7790 | 0.7799
    5     | 0.6241        | 0.7061          | 0.7955   | 0.8013    | 0.7955 | 0.7967
    6     | 0.5303        | 0.6782          | 0.7945   | 0.8009    | 0.7945 | 0.7963
    ```
    - GPU Memory Post-Training: 334.84 MB.
  - **Test Evaluation**:
    ```
    Test Results:
    - Eval Loss: 0.7024
    - Accuracy: 0.7955 (79.55%)
    - Precision: 0.8025
    - Recall: 0.7955
    - F1: 0.7971
    - Runtime: ~228 seconds (3.8 minutes)
    ```
    - **Financial Class Accuracy**:
      - Form (label 1, tax forms proxy): 68.00%.
      - Budget (label 10, financial reports proxy): 69.60%.
      - Invoice (label 11): 77.60%.
      - Memo (label 15, financial reports proxy): 76.00%.
  - **Model Size**: 106 MB on disk.

## Improvements
- **Data Scaling**: Increased from 5,000 to 20,000 samples, lifting accuracy from 59.15% to 78.85% (ViT) and 79.55% (Swin).
- **Balancing**: Post-5,000, balanced sampling corrected class skew, improving fairness and financial class accuracy (e.g., invoice: 68% → 72.8% ViT, 77.6% Swin).
- **Augmentation**: Enhanced robustness to document variations.
- **Tuning**: Optimized learning rate, batch size, and early stopping for stability and performance.
- **Model Exploration**: Swin-Tiny outperformed ViT-Base slightly (79.55% vs. 78.85%), with notable gains in financial classes due to layout sensitivity.

## Running the Code
1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   ```
2. **Run Notebooks**:
   - Open in Colab with GPU enabled.
   - Execute cells in `notebooks/rvl_cdip_vit_20000.ipynb` or `rvl_cdip_swin_20000.ipynb`.

## Repository Structure
```
rvl_cdip_classification/
├── notebooks/
│   ├── rvl_cdip_vit_5000.ipynb    # 59.15% accuracy
│   ├── rvl_cdip_vit_10000.ipynb   # 77.2% accuracy
│   ├── rvl_cdip_vit_20000.ipynb   # 78.85% accuracy
│   └── rvl_cdip_swin_20000.ipynb  # 79.55% accuracy
├── models/
│   ├── rvl_cdip_vit_model/        # 328 MB
│   └── rvl_cdip_swin_model/       # 106 MB
└── README.md
```

## Future Work
- Scale to the full 320,000-sample dataset with paid resources.
- Test layout-aware models (e.g., LayoutLM) for text-heavy documents.

## Contributors
- Prasad Gavhane
