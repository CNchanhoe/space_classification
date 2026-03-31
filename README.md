# Stellar Classification using Deep Neural Networks (DNN)

This repository contains a PyTorch implementation of a Deep Neural Network (DNN) to classify astronomical objects into three categories: **GALAXY**, **STAR**, and **QSO** (Quasar), based on the SDSS (Sloan Digital Sky Survey) dataset.

## 🚀 Project Overview
The goal of this project is to automate the classification of celestial bodies using photometric data and redshift. The model achieves high accuracy by learning the spectral characteristics of different stellar objects.

## 📊 Dataset
The project uses the `star_classification.csv` dataset. 
- **Observations**: 100,000
- **Input Features**: 8 features (after preprocessing), including $u, g, r, i, z$ filters and `redshift`.
- **Target Classes**: 
  - `GALAXY`
  - `STAR`
  - `QSO` (Quasar)

*Note: Technical identifiers like `obj_ID`, `run_ID`, etc., are dropped during preprocessing as they do not carry physical information.*

## 🧠 Model Architecture
The model is a Multi-Layer Perceptron (MLP) built with **PyTorch**:
- **Input Layer**: Number of features (after scaling)
- **Hidden Layer 1**: 128 units + ReLU + Dropout (0.3)
- **Hidden Layer 2**: 64 units + ReLU + Dropout (0.2)
- **Hidden Layer 3**: 32 units + ReLU
- **Output Layer**: 3 units (Classification)

## 🛠️ Features
- **Early Stopping**: Prevents overfitting by monitoring validation loss (patience = 10).
- **Standardization**: Uses `StandardScaler` for optimal neural network performance.
- **Model Saving**: Automatically saves the best model state as `best_dnn_model.pth`.
- **Performance Evaluation**: Provides a full classification report (Precision, Recall, F1-Score).

## 💻 Installation & Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch pandas scikit-learn numpy
