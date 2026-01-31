"""
Preprocessing module (owned by Preprocessing owner).
Cleans raw data: handle special codes, type normalization.
Exposes: clean(df) -> cleaned DataFrame
"""
# TODO: Integrate teammate's preprocess.clean() when merged
# Placeholder: pass-through until real implementation is available

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw DataFrame.
    Handles special codes (9, 99, 6 -> missing), basic type normalization.
    """
    # Placeholder: return as-is until preprocessing owner merges
    return df.copy()
