import pandas as pd
from typing import List, Tuple

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # TEMP: no engineering yet; replace with your real feature code later
    return df.copy()

def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Treat non-numeric columns as categorical, numeric as numeric
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, num_cols

