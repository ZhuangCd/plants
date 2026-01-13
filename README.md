# Plant Disease Classification Using Vision Transformers

A deep learning approach to classify plant diseases from leaf images using Vision Transformer (ViT) models with explainability analysis.

---

## Project Overview

This is a machine learning final project that tackles plant disease classification using the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle. The project implements Vision Transformer (ViT) models for image classification and includes model interpretability analysis using DeepSHAP.



---

## Methodology

### 1. Exploratory Data Analysis
- Dataset exploration and visualization
- Class distribution analysis
- Image quality assessment

### 2. Model Architecture
- **Vision Transformer (ViT)** using `timm` library
- Pre-trained weights with fine-tuned classification head
- Input size: 224x224 pixels
- Patch size: 16x16 pixels

### 3. Training Strategy
- **Phase 1:** Train only classification head (frozen backbone)
- **Phase 2:** Fine-tune entire model
- Addressed class imbalance using:
  - Weighted loss function
  - WeightedRandomSampler
  - Data augmentation (Color Jitter, Cutout, Mixup)

### 4. Evaluation Metrics
- Accuracy
- F1 Score
- Confusion Matrix

### 5. Model Interpretability
- DeepSHAP analysis for feature attribution
- Grad-cam

---

## Libraries

- **Python**
- **PyTorch** - Deep learning framework
- **timm** - PyTorch Image Models library
- **SHAP** - Model explainability
- **Kaggle Hub** - Dataset access
- **Matplotlib / Seaborn** - Visualization

---

## Getting Started

### Prerequisites
```bash
pip install torch torchvision timm kagglehub shap matplotlib pandas tqdm
```

### Download Dataset
```python
import kagglehub
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
```

### Run Notebooks
1. Start with `notebooks/EDA_plants.ipynb` for data exploration
2. Train models using `notebooks/ViT.ipynb` or `notebooks/ViT_all_15epoch.ipynb`
3. Analyze model predictions with `notebooks/DeepSHAP_Analysis.ipynb`

