# 🎯 Graduate Underemployment / Overqualification Prediction — ML Pipeline

This repository implements a **full machine learning pipeline** for predicting **overqualification** (underemployment) in recruitment using the **NGS (National Graduate Survey) structured hiring dataset**. It was developed in the context of the **ML Hackathon hosted by the SFU Data Science Student Society**, where teams worked with real-world datasets to design, train, and evaluate predictive models in a competitive setting.

The pipeline uses **CatBoost** as the primary model, with a focus on **predictive performance** (accuracy on Public/Private leaderboards) and **interpretability** (feature importance and optional SHAP).

---

## 🎯 Project Overview

The goal of this project is to:

- **Build a robust model** that accurately estimates overqualification probability based on candidate attributes: education level, years of experience, skill composition, prior roles, and demographics.
- **Work with the NGS dataset** and understand its feature structure (survey codes, missing conventions, mixed-type columns).
- **Train and tune a CatBoost-based** machine learning model with validation feedback and leaderboard-oriented iteration.
- **Focus on both predictive performance and interpretability** — accuracy on hold-out test sets and feature importance / SHAP-style explanations.

The solution achieved **0.75174** accuracy on the Public leaderboard and **0.70511** on the Private leaderboard, placing it very close to the top-performing teams and demonstrating strong generalization on unseen data.

![EDA and feature exploration](./submissions/exploration%20heatmap.png)
_Exploratory analysis: correlation and feature relationships in the NGS hiring dataset_

---

## ✨ Key Features

- **Modular ML pipeline** (`src/` folder): clean separation of data loading, preprocessing, feature engineering, model training, evaluation, and prediction.
- **NGS-aware preprocessing:** handling of special codes (6, 9, 99) and normalization of mixed-type columns (e.g. GENDER2, DDIS_FL, VISBMINP).
- **CatBoost classifier** with native categorical support, early stopping, and configurable hyperparameters (depth, learning_rate, l2_leaf_reg).
- **Stratified K-fold cross-validation** and optional grid search for hyperparameter tuning.
- **Interpretability:** CatBoost feature importance and optional SHAP integration for model explanation.
- **Reproducible workflow:** `python3 -m src.train` and `python3 -m src.predict` for end-to-end training and submission generation.
- **Five structured Jupyter notebooks** documenting exploration, preprocessing, training/tuning, evaluation/interpretability, and the full pipeline demo.

---

## 🧱 Repository Structure

```
graduate-underemployment-prediction/
│
├── data/
│   ├── processed/                                  # Processed/cached data (optional); not in Git
│   └── raw/
│       ├── train.csv                               # Training set (id, features, overqualified)
│       └── test.csv                                # Test set (id, features; no target)
│
├── models/                                         # Saved model artifacts (model.cbm, artifacts.pkl); not in Git
│
├── notebooks/
│   ├── 01_exploration.ipynb                        # EDA, NGS feature structure, target and correlations
│   ├── 02_preprocessing_feature_engineering.ipynb  # Cleaning and categorical encoding
│   ├── 03_catboost_training_tuning.ipynb           # Training, CV, hyperparameter tuning
│   ├── 04_evaluation_interpretability.ipynb        # Metrics, feature importance, SHAP
│   └── 05_pipeline_demo.ipynb                      # End-to-end pipeline demonstration
│
├── submissions/                                    # Generated submission CSVs (id, overqualified)
│   ├── public_leaderboards.png                     # Public leaderboard screenshot
│   └── submission.csv                              # Default output from python3 -m src.predict
│
├── src/
│   ├── __init__.py
│   ├── config.py                                   # Paths, target/id columns, validation settings
│   ├── data.py                                     # Load train/test, split X/y, train/val split
│   ├── evaluate.py                                 # Stratified K-fold CV and accuracy
│   ├── features.py                                 # Categorical feature preparation for CatBoost
│   ├── hyperparameter_tuning.py                    # Grid search for CatBoost params
│   ├── model.py                                    # CatBoost classifier builder
│   ├── preprocess.py                               # NGS cleaning and categorical normalization
│   ├── predict.py                                  # Load model, predict on test, write submission
│   └── train.py                                    # End-to-end training pipeline
│
├── .gitignore                                      # Git ignore rules (venv, models/*, cache, etc.)
├── LICENSE                                         # MIT license
├── README.md                                       # Project overview and usage
├── report.md                                       # Detailed technical write-up
└── requirements.txt                                # Python dependencies
```

> 🗒️ **Note:**  
> The `data/raw/` directory should contain `train.csv` and `test.csv`. The `models/` directory is where the trained CatBoost model and artifacts are saved after running `python3 -m src.train`, **`models/` is not tracked in Git** (it is in `.gitignore`), so you need to run the training pipeline locally to generate the model. Processed data is not stored on disk; all transformations are applied in memory during training and prediction.

---

## 🧰 Run Locally

You can run this project on your machine using **Python 3.11+** and a virtual environment.

### 1️⃣ Clone the repository

**HTTPS (recommended for most users):**
```bash
git clone https://github.com/florykhan/graduate-underemployment-prediction.git
cd graduate-underemployment-prediction
```

**SSH (for users who have SSH keys configured):**
```bash
git clone git@github.com:florykhan/graduate-underemployment-prediction.git
cd graduate-underemployment-prediction
```

### 2️⃣ Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add the dataset

Place the NGS hackathon data files in `data/raw/`:

```
data/raw/train.csv   # Training set (must include column: overqualified)
data/raw/test.csv    # Test set (same features, no target)
```

> 📥 **Dataset:** The NGS structured hiring dataset was provided as part of the SFU Data Science Student Society ML Hackathon. Ensure `train.csv` has an `id` column and an `overqualified` (0/1) target column; `test.csv` should have the same feature columns and `id`.

### 5️⃣ Run the training pipeline

This step trains the CatBoost model, runs validation (and optional CV), and saves the model and artifacts.

```bash
python3 -m src.train
```

### 6️⃣ Generate predictions and submission

```bash
python3 -m src.predict
```

This writes `submissions/submission.csv` with columns `id` and `overqualified` (0/1 predictions).

### 7️⃣ Run the notebooks

Launch Jupyter and open the notebooks from the project root (so that `notebooks/` is the working directory for paths):

```bash
jupyter notebook
```

Recommended order:

- `notebooks/01_exploration.ipynb` — data exploration and NGS feature structure
- `notebooks/02_preprocessing_feature_engineering.ipynb` — cleaning and categorical encoding
- `notebooks/03_catboost_training_tuning.ipynb` — CatBoost training, CV, and tuning
- `notebooks/04_evaluation_interpretability.ipynb` — metrics, feature importance, SHAP
- `notebooks/05_pipeline_demo.ipynb` — end-to-end pipeline demo

> **Tip:** If you run notebooks from inside `notebooks/`, the code uses `sys.path.insert(0, str(Path().resolve().parent))` so that `src` can be imported correctly.

---

## 📊 Results (Summary)

The hackathon had **14 teams** in total. Public leaderboard snapshot:

![Public leaderboard](submissions/public_leaderboards.png)

| **Metric** | **Value** |
|------------|-----------|
| Public leaderboard accuracy | **0.75174** (best: 0.76623) |
| Private leaderboard accuracy | **0.70511** (best: 0.71304) |

The tuned CatBoost model placed the solution very close to the top-performing teams and demonstrated strong generalization on the private hold-out set. Validation and cross-validation accuracy (e.g. ~0.67–0.75 depending on split and hyperparameters) are used during development; the leaderboard metrics above reflect the official hackathon evaluation.

➡️ For methodology, preprocessing details, model choices, and full discussion, see: [`report.md`](report.md).

---

## 📄 Full Technical Report

The complete technical write-up, including pipeline design, preprocessing and feature engineering, CatBoost training and tuning, validation strategy, and interpretability, is in [`report.md`](report.md). This document is intended for reviewers who want the full methodology behind the pipeline and results.

---

## 🚀 Future Directions

- **Expand hyperparameter search:** use RandomizedSearchCV or Optuna over a larger CatBoost parameter space.
- **Feature engineering:** additional derived features (e.g. education–occupation match indicators) if metadata is available.
- **Ensembles:** combine CatBoost with other classifiers (e.g. XGBoost, LightGBM) for potential accuracy gains.
- **Experiment tracking:** integrate MLflow or Weights & Biases to log metrics and compare runs.
- **Production readiness:** API (FastAPI/Flask), Docker, or CI/CD for training and deployment.

---

## 🧠 Tech Stack

- **Language:** Python 3.11+
- **Core libraries:** pandas, numpy, scikit-learn, CatBoost, matplotlib, seaborn
- **Pipeline:** Modular `src/` package with config, data loading, preprocessing, feature engineering, model, evaluation, tuning, train, and predict
- **Environment:** Jupyter Notebook / VS Code; Git

---

## 🧾 License

MIT License, feel free to use and modify with attribution. See the [`LICENSE`](LICENSE) file for full details.

---

## 👤 Authors

**Ilian Khankhalaev**  
_BSc Computing Science, Simon Fraser University_  
📍 Vancouver, BC  |  [florykhan@gmail.com](mailto:florykhan@gmail.com)  |  [GitHub](https://github.com/florykhan)  |  [LinkedIn](https://www.linkedin.com/in/ilian-khankhalaev/)

**Nikolay Deinego**  
_BSc Computing Science, Simon Fraser University_  
📍 Vancouver, BC  | [GitHub](https://github.com/Deinick)  |  [LinkedIn](https://www.linkedin.com/in/nikolay-deinego/)

**Arina Veprikova**  
_BSc Data Science, Simon Fraser University_  
📍 Vancouver, BC  |  [GitHub](https://github.com/areenve)  |  [LinkedIn](https://www.linkedin.com/in/arina-veprikova-a97526366/)

**Anna Cherkashina**  
_BSc Data Science, Simon Fraser University_  
📍 Vancouver, BC  | [GitHub](https://github.com/Anna05072005)  |  [LinkedIn](https://www.linkedin.com/in/anna-cherkashina-467059293/)
