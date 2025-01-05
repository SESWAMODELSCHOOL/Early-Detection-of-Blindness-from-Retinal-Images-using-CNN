# Early Detection of Blindness from Retinal Images Using CNN

This project develops a deep learning-based solution for early detection of diabetic retinopathy (DR) using retinal fundus images. The fusion-based CNN model combines EfficientNetB0 and InceptionV3 architectures for efficient and accurate classification. The results include performance from hyperparameter tuning and various experiments conducted during the study.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Hyperparameter Tuning Results](#hyperparameter-tuning-results)

---

## Overview

Diabetic retinopathy (DR) is a leading cause of blindness worldwide, requiring early detection to prevent severe vision loss. This project leverages advanced deep learning techniques to classify retinal fundus images into DR severity levels. A Streamlit-based web application is included for real-time interaction.

---

## Features
- **Fusion Model**: Combines EfficientNetB0 and InceptionV3 for robust feature extraction and classification.
- **Multi-Class and Binary Classification**: Classifies images into "No DR," "Mild DR," and "Severe DR" or performs binary classification for specific cases.
- **Streamlit App**: Provides an intuitive interface for uploading images and receiving predictions.
- **Hyperparameter Tuning**: Optimized dropout rates, dense units, and learning rates for the best model performance.
- **Evaluation Metrics**: Includes accuracy, precision, recall, F1-score, confusion matrices, and ROC curves.

---

## Dataset
The dataset is sourced from the [APTOS 2019 Blindness Detection Challenge](https://www.kaggle.com/c/aptos2019-blindness-detection), consisting of labeled retinal fundus images for various DR severity levels.

---

## Project Structure
```
├── App/
│   ├── test_images/
│   ├── app.py             # Streamlit application for predictions
│   ├── req.txt            # Required Python libraries
├── Code/
│   ├── 00) diabetic-retinopathy-detection.ipynb   # Multi-class classification notebook
│   ├── 01) Finetuning-binary(No DR vs Severe).ipynb # Binary classification notebook
├── LICENSE                # License file
├── Model Architecture.png # Visualization of the model architecture
└── README.md              # Project documentation
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SESWAMODELSCHOOL/Early-Detection-of-Blindness-from-Retinal-Images-using-CNN.git
   cd Early-Detection-of-Blindness-from-Retinal-Images-using-CNN
   ```

2. Install dependencies:
   ```bash
   pip install -r App/req.txt
   ```

3. Download and extract the APTOS 2019 dataset from Kaggle and place it in the `App/test_images` folder.

---

## Usage

### Streamlit Web App
Run the Streamlit application to make predictions on test images:
```bash
streamlit run App/app.py
```

### Notebooks
Use the provided Jupyter Notebooks for training and evaluating models:
- **Multi-Class Classification**: `Code/00) diabetic-retinopathy-detection.ipynb`
- **Binary Classification**: `Code/01) Finetuning-binary(No DR vs Severe).ipynb`

---

## Results

### Original Model Results
| Experiment                          | Accuracy | Precision | Recall | F1 Score | AUC  |
|-------------------------------------|----------|-----------|--------|----------|------|
| Multi-Class Classification          | 84.44%   | 83.74%    | 84.44% | 82.15%   | 0.80 |
| Binary Classification (No DR vs DR) | 95.77%   | 95.79%    | 95.77% | 95.77%   | 0.96 |
| No DR vs Severe                     | 97.62%   | 97.62%    | 97.62% | 97.62%   | 0.95 |
| Balanced Data Modeling              | 33.33%   | 11.11%    | 33.33% | 16.67%   | 0.50 |

### Best Model Results (After Hyperparameter Tuning)
- **Accuracy**: 95%
- **Precision**: 95.27%
- **Recall**: 95%
- **F1 Score**: 94.53%

---

## Hyperparameter Tuning Results
|    | Dropout Rate | Dense Units | Learning Rate | Accuracy | Precision | Recall   | F1 Score |
|----|--------------|-------------|---------------|----------|-----------|----------|----------|
|  0 | 0.3          | 256         | 0.0001        | 90.71%   | 1.00      | 33.90%   | 50.63%   |
|  1 | 0.4          | 128         | 0.0005        | 91.67%   | 1.00      | 40.68%   | 57.83%   |
|  2 | 0.4          | 128         | 0.0005        | 91.43%   | 0.96      | 40.68%   | 57.14%   |
|  3 | 0.5          | 256         | 0.001         | 95.00%   | 1.00      | 64.41%   | 78.35%   |
|  4 | 0.5          | 256         | 0.001         | 91.90%   | 93.10%    | 45.76%   | 61.36%   |

**Best Parameters**:
- **Dropout Rate**: 0.5
- **Dense Units**: 256
- **Learning Rate**: 0.001
