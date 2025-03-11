# RVL-CDIP Document Classification System

## Overview
This project develops a transformer-based document classification system for the RVL-CDIP dataset, focusing on financial and business documents (e.g., invoices, forms, budgets, memos). Built using Vision Transformer (ViT) and fine-tuned on subsets of the dataset, the model achieves high accuracy within Colab/Kaggle free-tier constraints.

## Assignment Requirements
- **Objective**: Classify RVL-CDIP images into 16 categories, emphasizing financial types (invoices, receipts, tax forms, financial reports).
- **Dataset**: `aharley/rvl_cdip` (400,000 images, 16 classes).
- **Progress**:
  - 5,000 samples: 59.15% accuracy.
  - 10,000 samples: 77.2% accuracy.
  - 20,000 samples: Ongoing (expected ~80-85%).

## Model Selection
- **Why ViT**: 
  - Pre-trained on ImageNet-21k, effective for image classification.
  - Transformer architecture excels at capturing global context in document layouts.
- **Base Model**: `google/vit-base-patch16-224-in21k`.

## Methodology
- **Data**: Streamed subsets (5,000 → 20,000) from 320,000 training images due to memory limits.
- **Fine-Tuning**:
  - Balanced sampling for equal class representation.
  - Data augmentation (rotation, jitter) for generalization.
  - Learning rate: 5e-5 → 3e-5.
  - Batch size: 32 → 64 (via gradient accumulation).
  - Early stopping (patience=2).
- **Constraints**: Colab T4 GPU (15 GB VRAM, 12.5 GB RAM), ~12-hour runtime.

## Results
- **10,000 Samples**:
  - Overall Accuracy: 77.2%.
  - Financial Classes:
    - Form: 64.8%.
    - Budget: 67.2%.
    - Invoice: 68.0%.
    - Memo: 72.8%.
- **20,000 Samples**: Pending (see `rvl_cdip_20000.ipynb`).

## Improvements
- Increased data size (5,000 → 20,000).
- Added augmentation to boost financial class performance.
- Optimized hyperparameters (learning rate, epochs).

## Running the Code
1. Clone repo: `git clone <repo-url>`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open notebooks in Colab/Kaggle with GPU enabled.
4. Run cells sequentially.

## Example Predictions
(Added in `rvl_cdip_20000.ipynb`, Step 9 - pending results.)

## Future Work
- Full dataset training with paid resources.
- Upload model to Hugging Face Hub.

## Contributors
- [Your Name]
