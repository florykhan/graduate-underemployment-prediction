"""
Preprocessing module for the NGS hiring / overqualification dataset.
Cleans raw data: handle special codes (6, 9, 99 = missing/refused), normalize mixed types
Exposes: clean(df) -> cleaned DataFrame
"""
import numpy as np
import pandas as pd

# NGS convention: 6 = valid skip, 9 / 99 = refused / don't know / not stated → treat as missing
MISSING_CODES = {6.0, 9.0, 99.0, 6, 9, 99}


def _normalize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns that have mixed numeric and string values (e.g. GENDER2, DDIS_FL, VISBMINP).
    Map common text values to consistent codes so they can be treated as categorical.
    """
    df = df.copy()

    # GENDER2: "Male" -> 1, "Female" -> 2, numeric keep as-is (1/2/9 etc.)
    if "GENDER2" in df.columns:
        g = df["GENDER2"].astype(str).str.strip().str.lower()
        g = g.replace({"male": "1", "female": "2"})
        df["GENDER2"] = pd.to_numeric(g, errors="coerce")

    # DDIS_FL: "With disability" -> 1, "Without disability" -> 2
    if "DDIS_FL" in df.columns:
        d = df["DDIS_FL"].astype(str).str.strip().str.lower()
        d = d.replace({"with disability": "1", "without disability": "2"})
        df["DDIS_FL"] = pd.to_numeric(d, errors="coerce")

    # VISBMINP: "Yes" -> 1, etc. (survey often uses 1/2 for yes/no)
    if "VISBMINP" in df.columns:
        v = df["VISBMINP"].astype(str).str.strip().str.lower()
        v = v.replace({"yes": "1", "no": "2"})
        df["VISBMINP"] = pd.to_numeric(v, errors="coerce")

    return df


def _replace_missing_codes(df: pd.DataFrame, code_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Replace NGS missing/refused codes (6, 9, 99) with np.nan in numeric/code columns.
    """
    df = df.copy()
    if code_columns is None:
        # All feature columns that are numeric (excluding id and target)
        code_columns = [
            c for c in df.columns
            if c not in ("id", "overqualified") and df[c].dtype in (np.floating, np.integer)
        ]
    for col in code_columns:
        if col not in df.columns:
            continue
        if df[col].dtype in (np.floating, np.integer):
            df[col] = df[col].replace(list(MISSING_CODES), np.nan)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw NGS DataFrame.
    - Normalizes mixed-type categorical columns (GENDER2, DDIS_FL, VISBMINP).
    - Replaces special codes (6, 9, 99) with NaN.
    - Keeps id and overqualified (if present) unchanged.
    """
    df = df.copy()
    df = _normalize_categorical_columns(df)
    df = _replace_missing_codes(df)
    return df
