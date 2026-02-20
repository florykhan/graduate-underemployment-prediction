"""
Hyperparameter tuning for CatBoost: grid or random search over key parameters.
Used for leaderboard-oriented iteration (e.g. depth, learning_rate, l2_leaf_reg)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from src.model import build_model


def grid_search_cv(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict,
    cat_indices: list[int],
    n_folds: int = 5,
    random_state: int = 42,
    early_stopping_rounds: int = 20,
) -> list[dict]:
    """
    Simple grid search over param_grid; each key is a list of values.
    Returns list of dicts: {params, mean_accuracy, std_accuracy}.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    def _expand(prefix, idx):
        if idx == len(keys):
            yield prefix
            return
        for v in values[idx]:
            yield from _expand(prefix + [(keys[idx], v)], idx + 1)

    results = []
    for param_list in _expand([], 0):
        params = dict(param_list)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = build_model(
                verbose=0,
                early_stopping_rounds=early_stopping_rounds,
                random_seed=random_state,
                **params,
            )
            model.fit(
                X_train, y_train,
                cat_features=cat_indices,
                eval_set=(X_val, y_val),
            )
            pred = model.predict(X_val)
            fold_scores.append(accuracy_score(y_val, pred))
        results.append({
            "params": params,
            "mean_accuracy": float(np.mean(fold_scores)),
            "std_accuracy": float(np.std(fold_scores)),
        })
    return results
