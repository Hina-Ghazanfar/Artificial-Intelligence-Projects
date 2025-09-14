# Student Analysis — Foundations of AI

> Predicting student grades with exploratory data analysis (EDA), preprocessing, and a baseline **Linear Regression** model on the classic `student-mat.csv` dataset.

![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![License](https://img.shields.io/badge/License-Your%20Choice-lightgrey)

## Overview
This repository contains a single Jupyter notebook, **`Student Analysis-Foundations of AI.ipynb`**, that walks through a compact, end‑to‑end applied ML workflow:
- Load and inspect the **UCI Student Performance (Math)** dataset (`student-mat.csv`).
- Perform **EDA** (distributions, correlations, pair plots, missing‑value checks).
- Apply **preprocessing** (label encoding for categoricals, standardization for numeric features).
- Train a **Linear Regression** model to predict the final grade **G3** (with references to **G1/G2** as features).
- Evaluate using **R²** and **MSE**, and visualize results.

> Why this notebook? It’s a clean, classroom‑quality example you can extend with regularization (Ridge/Lasso), tree ensembles, cross‑validation, and feature‑importance analysis.

## Repository Structure
```
.
├── Student Analysis-Foundations of AI.ipynb
├── data/
│   └── student-mat.csv              # Place the dataset here
├── requirements.txt                 # Minimal runtime dependencies
└── README.md
```

## Getting Started

### 1) Clone and set up a virtual environment
```bash
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>

# Create and activate a virtual environment (choose one)
python -m venv .venv && source .venv/bin/activate   # macOS/Linux
# or:
py -m venv .venv && .venv\Scripts\activate        # Windows PowerShell

pip install -r requirements.txt
```

### 2) Obtain the dataset
Download **`student-mat.csv`** (UCI Student Performance – Mathematics) and place it at:
```
data/student-mat.csv
```
> Tip: If your copy lives elsewhere, update the path in the notebook’s data‑loading cell.

### 3) Run the notebook
```bash
jupyter lab
# or
jupyter notebook
```
Open **`Student Analysis-Foundations of AI.ipynb`** and run the cells top‑to‑bottom.

## What’s Inside the Notebook
- **EDA**
  - Summary stats, missing‑value checks
  - Histograms/distributions and scatter plots
  - **Correlation heatmap** and selected pair plots
- **Preprocessing**
  - `LabelEncoder` for categorical columns
  - `StandardScaler` for numeric features
  - Train/validation split with `train_test_split`
- **Modeling**
  - Baseline **LinearRegression** (scikit‑learn)
- **Evaluation**
  - **R²** and **MSE** (optionally compute RMSE = √MSE)
  - Simple diagnostic plots

## Results (Baseline)
The notebook computes **R²** and **MSE** for the Linear Regression baseline. Exact values depend on your train/validation split. Use these as reference points while you iterate with stronger models.

## Extend This Project
- **Regularization:** Ridge / Lasso to reduce overfitting.
- **Non‑linear models:** RandomForestRegressor, Gradient Boosting, or XGBoost.
- **Cross‑validation:** `cross_val_score` for more stable estimates.
- **Feature engineering:** Domain‑driven feature selection or interaction terms.
- **Model comparison:** A small leaderboard table (R² / RMSE) across models.
- **Explainability:** Coefficient magnitudes, permutation importance, PDPs/ICE.

### Quick Example: Ridge/Lasso
```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

model = Ridge(alpha=1.0)  # try Lasso(alpha=0.1)
model.fit(X_train, y_train)
pred = model.predict(X_val)

print("R²:", r2_score(y_val, pred))
print("MSE:", mean_squared_error(y_val, pred))
```

## Requirements
Core dependencies identified from the notebook:
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `scipy`
- `jupyter`

Install everything via:
```bash
pip install -r requirements.txt
```

## Reproducibility Notes
- Results will vary slightly across runs due to the **train/test split** randomization.
- Set `random_state` in `train_test_split` and any models to make results reproducible.

## Dataset Citation
If you publish results, please cite the original dataset/paper (commonly referenced as):
> Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance.*

## License
Choose a license for your repository (e.g., MIT, Apache‑2.0) and update the badge at the top.

## Acknowledgements
- UCI Machine Learning Repository for the Student Performance dataset.
- scikit‑learn, pandas, numpy, matplotlib, and seaborn communities.

---

**Maintainer tips:** Keep the notebook tidy (clear outputs before commit), consider exporting key plots to `figures/`, and pin package versions in `requirements.txt` for stable builds.
