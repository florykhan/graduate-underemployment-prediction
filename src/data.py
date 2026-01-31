"""
Data loading utilities.
Loads raw CSVs and provides train/test splits with features and target.
"""
import pandas as pd

from src.config import (
    TRAIN_CSV,
    TEST_CSV,
    TARGET_COL,
    ID_COL,
    VAL_SIZE,
    RANDOM_STATE,
)


def load_train() -> pd.DataFrame:
    """Load raw train.csv as DataFrame."""
    return pd.read_csv(TRAIN_CSV)


def load_test() -> pd.DataFrame:
    """Load raw test.csv as DataFrame."""
    return pd.read_csv(TEST_CSV)


def split_X_y(df: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features X and target y.
    Excludes id and target columns from X.
    """
    cols_to_drop = [c for c in [ID_COL, target_col] if c in df.columns]
    X = df.drop(columns=cols_to_drop, errors="ignore")
    y = df[target_col] if target_col in df.columns else None
    return X, y


def get_train_val_split(df: pd.DataFrame, val_size: float = VAL_SIZE, random_state: int = RANDOM_STATE):
    """
    Split train DataFrame into train/val by row index.
    Returns (train_df, val_df).
    """
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=random_state, stratify=df[TARGET_COL])
    return train_df, val_df
