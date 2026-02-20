"""
Feature engineering for overqualification prediction.
Adds derived features and prepares feature set for CatBoost (handles categoricals natively).
Exposes: add_features(df) -> DataFrame with engineered features; get_feature_columns() for list of names
"""
import pandas as pd
import numpy as np

from src.config import TARGET_COL, ID_COL

# Columns that are categorical (survey codes): use as object/category for CatBoost
CATEGORICAL_FEATURES = [
    "CERTLEVP", "PGMCIPAP", "PGM_P034", "PGM_P036", "PGM_280A", "PGM_280B", "PGM_280C", "PGM_280F",
    "PGM_P401", "STULOANS", "DBTOTGRD", "SCHOLARP", "PREVLEVP", "HLOSGRDP", "GRADAGEP",
    "GENDER2", "CTZSHIPP", "VISBMINP", "DDIS_FL", "PAR1GRD", "PAR2GRD", "BEF_P140", "BEF_160",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features and ensure categorical columns have consistent type for CatBoost.
    - Converts code columns to string (CatBoost treats object as categorical).
    - Fills NaN with "missing" for categoricals.
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        # Coerce to numeric then to string so mixed types (e.g. "Female") become consistent
        try:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.fillna(-1).astype(int).astype(str)
        except Exception:
            df[col] = df[col].fillna("missing").astype(str)
        df.loc[df[col] == "-1", col] = "missing"
    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used for modeling (excluding id and target)."""
    return [c for c in CATEGORICAL_FEATURES if True]  # all we use for now


def get_categorical_feature_names() -> list[str]:
    """Return column names that should be passed to CatBoost as categorical."""
    return list(CATEGORICAL_FEATURES)
