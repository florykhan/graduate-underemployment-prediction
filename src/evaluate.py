"""
Evaluation module: stratified K-fold cross-validation with accuracy (and optional F1).
Uses CatBoost with early stopping on each fold.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.model import build_model


def run_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
    n_folds: int = 5,
    random_state: int = 42,
    cat_indices: list[int] | None = None,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    early_stopping_rounds: int = 20,
) -> dict:
    """
    Run stratified K-fold cross-validation with CatBoost.
    Returns dict with mean_accuracy, std_accuracy, fold_scores (accuracy per fold).
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_accuracies = []
    fold_f1 = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = build_model(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_state,
            early_stopping_rounds=early_stopping_rounds,
            verbose=0,
        )
        fold_model.fit(
            X_train, y_train,
            cat_features=cat_indices or [],
            eval_set=(X_val, y_val),
        )
        pred = fold_model.predict(X_val)
        fold_accuracies.append(accuracy_score(y_val, pred))
        fold_f1.append(f1_score(y_val, pred, average="macro"))

    return {
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "fold_accuracies": fold_accuracies,
        "mean_f1_macro": float(np.mean(fold_f1)),
        "std_f1_macro": float(np.std(fold_f1)),
    }
