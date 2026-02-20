"""
Training pipeline: preprocess, feature engineering, CatBoost training, validation, and artifact saving
1) Train on train.csv with optional validation split
2) Tune / validate using stratified K-fold and accuracy
3) Retrain on full train.csv
4) Save model and artifacts for predict.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

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
from src.features import add_features, get_categorical_feature_names
from src.model import build_model
from src.evaluate import run_validation

MODEL_FILE = "model.cbm"
ARTIFACTS_FILE = "artifacts.pkl"


def _get_cat_indices(X: pd.DataFrame, cat_names: list[str]) -> list[int]:
    """Return indices of columns in X that are in cat_names."""
    return [i for i, c in enumerate(X.columns) if c in cat_names]


def run_train_pipeline(
    validate: bool = True,
    val_size: float = VAL_SIZE,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    early_stopping_rounds: int = 20,
) -> dict:
    """
    Full training pipeline:
    1. Load and preprocess train data
    2. Optionally run validation (train/val split + CV)
    3. Retrain on full train
    4. Save CatBoost model and artifacts
    Returns metrics dict (val_accuracy, cv_mean_accuracy, etc.).
    """
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_train()
    df = clean(df)
    df = add_features(df)

    X, y = split_X_y(df, target_col=TARGET_COL)
    if y is None:
        raise ValueError("Train data must have 'overqualified' column")
    y = y.astype(int)

    cat_names = [c for c in get_categorical_feature_names() if c in X.columns]
    cat_indices = _get_cat_indices(X, cat_names)

    metrics = {}

    if validate:
        train_df, val_df = get_train_val_split(df, val_size=val_size, random_state=random_state)
        X_train, y_train = split_X_y(train_df, target_col=TARGET_COL)
        X_val, y_val = split_X_y(val_df, target_col=TARGET_COL)
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)

        cv_results = run_validation(
            X_train, y_train,
            model=None,
            n_folds=n_folds,
            random_state=random_state,
            cat_indices=cat_indices,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            early_stopping_rounds=early_stopping_rounds,
        )
        metrics["cv_mean_accuracy"] = cv_results["mean_accuracy"]
        metrics["cv_std_accuracy"] = cv_results["std_accuracy"]

        model = build_model(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            X_train, y_train,
            cat_features=cat_indices,
            eval_set=(X_val, y_val),
        )
        val_pred = model.predict(X_val)
        from sklearn.metrics import accuracy_score
        metrics["val_accuracy"] = float(accuracy_score(y_val, val_pred))
        print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
        print(f"CV accuracy: {metrics['cv_mean_accuracy']:.4f} ± {metrics['cv_std_accuracy']:.4f}")

    model = build_model(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(X, y, cat_features=cat_indices)

    artifacts = {
        "feature_cols": X.columns.tolist(),
        "cat_feature_names": cat_names,
        "target_col": TARGET_COL,
        "id_col": ID_COL,
    }
    model.save_model(str(MODEL_ARTIFACT_DIR / MODEL_FILE))
    import pickle
    with open(MODEL_ARTIFACT_DIR / ARTIFACTS_FILE, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Model saved to {MODEL_ARTIFACT_DIR / MODEL_FILE}")

    return metrics


if __name__ == "__main__":
    run_train_pipeline(validate=True)
