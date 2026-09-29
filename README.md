# Fashion-MNIST ANN Pipeline with Git, DVC & Google Drive

## Project Overview
This repository contains an end-to-end Machine Learning pipeline for classifying the **Fashion-MNIST** dataset using a fully-connected Artificial Neural Network (ANN) built with TensorFlow/Keras. The project demonstrates enterprise-grade MLOps practices:
- **Version Control**: Git workflow covering branching, rebasing, stashing, resets, and conflict resolution.
- **Data & Model Versioning**: Data Version Control (DVC) linked to Google Drive as remote storage.
- **Reproducibility**: DVC pipeline (`dvc.yaml`) parametrized via `params.yaml` achieving $\ge 85\%$ test accuracy.

## Architecture & Project Structure
```text
fashion-ann-pipeline/
├── .gitignore             # Git ignore rules for virtualenvs, caches, DVC payloads
├── README.md              # Project documentation
├── params.yaml            # Central hyperparameters for preprocessing & training
├── dvc.yaml               # Declarative multi-stage pipeline definition
├── dvc.lock               # Pipeline execution lockfile with artifact hashes
├── metrics.json           # Evaluation metrics (test loss, accuracy)
├── src/
│   ├── prepare.py         # Downloads raw Fashion-MNIST into data/raw/
│   ├── preprocess.py      # Normalizes images and creates validation split
│   ├── train.py           # Trains Sequential ANN and outputs model & history
│   └── evaluate.py        # Generates metrics and confusion matrix
├── data/
│   ├── raw/               # Raw numpy arrays tracked via DVC
│   └── processed/         # Normalized train/val/test splits tracked via DVC
└── models/
    ├── model.h5           # Trained ANN weights tracked via DVC
    ├── history.csv        # Epoch-by-epoch training metrics
    └── confusion_matrix.png # Evaluation visualization
```

## Setup & Quickstart
1. **Clone repository**:
   ```bash
   git clone <REPO_URL>
   cd fashion-ann-pipeline
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Reproduce Pipeline**:
   ```bash
   dvc repro
   ```
   *Note: Ensure Python 3.9+ and dependencies are active in your virtual environment.*
4. **Push / Pull Data Artifacts**:
   ```bash
   dvc push
   dvc pull
   ```

<!-- Hotfix: Ensure git config and python environment are active -->
