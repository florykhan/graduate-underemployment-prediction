"""
Prediction pipeline - Integration owner.
Loads model, predicts on test.csv, writes submission CSV.
"""
from pathlib import Path

import pandas as pd
import pickle

from src.config import SUBMISSIONS_DIR, TARGET_COL, ID_COL
from src.data import load_test, split_X_y
from src.preprocess import clean
from src.features import add_features
from src.train import _prepare_for_model, MODEL_ARTIFACT_DIR, MODEL_FILE, ARTIFACTS_FILE


def run_predict_pipeline(output_name: str = "submission.csv") -> str:
    """
    Load test data, preprocess, predict, write submission.
    Returns path to written submission file.
    """
    artifacts_path = MODEL_ARTIFACT_DIR / ARTIFACTS_FILE
    model_path = MODEL_ARTIFACT_DIR / MODEL_FILE
    if not artifacts_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found. Run train.py first.\n"
            f"Expected: {model_path} and {artifacts_path}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    # Load and preprocess test
    df = load_test()
    df = clean(df)
    df = add_features(df)

    # Keep ids for submission
    ids = df[ID_COL].values
    X, _ = split_X_y(df, target_col=TARGET_COL)

    X_arr = _prepare_for_model(X, fit=False, scaler=scaler, feature_cols=feature_cols)

    preds = model.predict(X_arr)

    # Build submission
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
