"""
Prediction pipeline: load CatBoost model and artifacts, predict on test.csv, write submission CSV
"""
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
import pickle

from src.config import SUBMISSIONS_DIR, TARGET_COL, ID_COL
from src.data import load_test, split_X_y
from src.preprocess import clean
from src.features import add_features
from src.train import MODEL_ARTIFACT_DIR, MODEL_FILE, ARTIFACTS_FILE


def _get_cat_indices(X: pd.DataFrame, cat_names: list[str]) -> list[int]:
    return [i for i, c in enumerate(X.columns) if c in cat_names]


def run_predict_pipeline(output_name: str = "submission.csv") -> str:
    """
    Load test data, preprocess, predict with CatBoost, write submission.
    Returns path to written submission file.
    """
    artifacts_path = MODEL_ARTIFACT_DIR / ARTIFACTS_FILE
    model_path = MODEL_ARTIFACT_DIR / MODEL_FILE
    if not artifacts_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found. Run train first: python3 -m src.train\n"
            f"Expected: {model_path} and {artifacts_path}"
        )

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    feature_cols = artifacts["feature_cols"]
    cat_names = artifacts.get("cat_feature_names", [])

    df = load_test()
    df = clean(df)
    df = add_features(df)

    ids = df[ID_COL].values
    X, _ = split_X_y(df, target_col=TARGET_COL)

    # Align to training columns
    for c in feature_cols:
        if c not in X.columns:
            X[c] = "missing"
    X = X[feature_cols]

    preds = model.predict(X)

    submission = pd.DataFrame({
        ID_COL: ids,
        TARGET_COL: preds.astype(int),
    })
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / output_name
    submission.to_csv(out_path, index=False)
    print(f"Submission written to {out_path} ({len(submission)} rows)")
    return str(out_path)


if __name__ == "__main__":
    run_predict_pipeline()
