"""
Training pipeline - Integration owner.
Wires preprocess, features, model, and evaluate together.
1) Train on train.csv
2) Validate using train/val split
3) Retrain on full train.csv
4) Save model and artifacts for predict.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODEL_ARTIFACT_DIR,
    TARGET_COL,
    ID_COL,
    VAL_SIZE,
    RANDOM_STATE,
    N_FOLDS,
)
from src.data import load_train, split_X_y, get_train_val_split
from src.preprocess import clean
from src.features import add_features
from src.model import build_model
from src.evaluate import run_validation


MODEL_FILE = "model.pkl"
ARTIFACTS_FILE = "artifacts.pkl"


def _prepare_for_model(X: pd.DataFrame, fit: bool = True, scaler=None, feature_cols=None):
    """
    Prepare DataFrame for sklearn: fill NaNs, encode categoricals, scale.
    When fit=True: returns (X_array, scaler, feature_cols).
    When fit=False: uses provided scaler and feature_cols to transform.
    """
    X = X.copy()
    # Fill numeric NaNs with median
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())
    # Fill remaining object NaNs with 'missing'
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna("missing").astype(str)
    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)
    if fit:
        feature_cols = X.columns.tolist()
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X)
        return X_arr, scaler, feature_cols
    else:
        # Align to training columns
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[feature_cols]
        X_arr = scaler.transform(X)
        return X_arr


def run_train_pipeline(
    validate: bool = True,
    val_size: float = VAL_SIZE,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Full training pipeline:
    1. Load and preprocess train data
    2. Optionally run validation (train/val split)
    3. Retrain on full train
    4. Save model and artifacts
    Returns metrics dict.
    """
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess
    df = load_train()
    df = clean(df)
    df = add_features(df)

    X, y = split_X_y(df, target_col=TARGET_COL)
    if y is None:
        raise ValueError("Train data must have 'overqualified' column")
    y = y.astype(int)

    metrics = {}

    # 2. Optional validation
    if validate:
        train_df, val_df = get_train_val_split(df, val_size=val_size, random_state=random_state)
        X_train, y_train = split_X_y(train_df, target_col=TARGET_COL)
        X_val, y_val = split_X_y(val_df, target_col=TARGET_COL)
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)

        X_train_arr, scaler, feature_cols = _prepare_for_model(X_train, fit=True)
        X_val_arr = _prepare_for_model(X_val, fit=False, scaler=scaler, feature_cols=feature_cols)

        model = build_model(random_state=random_state)
        cv_results = run_validation(
            pd.DataFrame(X_train_arr), pd.Series(y_train), model, n_folds=n_folds, random_state=random_state
        )
        metrics["cv_mean_f1_macro"] = cv_results["mean_f1_macro"]
        metrics["cv_std_f1_macro"] = cv_results["std_f1_macro"]

        model.fit(X_train_arr, y_train)
        from sklearn.metrics import f1_score
        val_pred = model.predict(X_val_arr)
        metrics["val_f1_macro"] = float(f1_score(y_val, val_pred, average="macro"))
        print(f"Validation F1 (macro): {metrics['val_f1_macro']:.4f}")
        print(f"CV F1 (macro): {metrics['cv_mean_f1_macro']:.4f} ± {metrics['cv_std_f1_macro']:.4f}")

    # 3. Retrain on full train
    X_arr, scaler, feature_cols = _prepare_for_model(X, fit=True)
    model = build_model(random_state=random_state)
    model.fit(X_arr, y)

    # 4. Save artifacts
    artifacts = {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "id_col": ID_COL,
    }
    with open(MODEL_ARTIFACT_DIR / MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_ARTIFACT_DIR / ARTIFACTS_FILE, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Model saved to {MODEL_ARTIFACT_DIR / MODEL_FILE}")

    return metrics


if __name__ == "__main__":
    run_train_pipeline(validate=True)
