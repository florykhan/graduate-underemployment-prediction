"""
Evaluation module (owned by Model/Cross-validation owner).
Runs cross-validation and reports metrics (F1, Macro F1).
Exposes: run_validation(...)
"""
# TODO: Integrate teammate's evaluate.run_validation() when merged
# Placeholder: simple stratified K-fold CV with macro F1

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd


def run_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Run stratified K-fold cross-validation.
    Returns dict with mean_f1, std_f1, and fold scores.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    return {
        "mean_f1_macro": float(scores.mean()),
        "std_f1_macro": float(scores.std()),
        "fold_scores": scores.tolist(),
    }
