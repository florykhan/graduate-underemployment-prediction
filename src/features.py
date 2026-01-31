import pandas as pd
import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------
# Conservative missing-code rules (only apply where we are confident)
# ---------------------------------------------------------------------
# Many NGS columns use: 9 / 99 = "Not stated"
# Some use: 6 = "Valid skip" but NOT always; we only treat 6 as missing
# for variables where the dictionary explicitly says 6 = valid skip.
MISSING_CODES = {
    # "Not stated" style
    "default": {9, 99},
    # Column-specific "valid skip" codes (treat as missing)
    "PGM_P034": {6, 9},
    "PGM_P036": {6, 9},
    "PGM_P401": {6, 9},
    "PGM_280A": {6, 9},
    "PGM_280B": {6, 9},
    "PGM_280C": {6, 9},
    "PGM_280F": {6, 9},
    "STULOANS": {6, 9},
    "DBTOTGRD": {6, 9},
    "SCHOLARP": {6, 9},
    "PAR1GRD": {6, 9},
    "PAR2GRD": {6, 9},
    # "Not stated"
    "CERTLEVP": {9},
    "PGMCIPAP": {99},
    "HLOSGRDP": {9},
    "PREVLEVP": {9},
    "BEF_P140": {9},
    "BEF_160": {99},
    "VISBMINP": {9},
}

# Columns that should remain categorical even if numeric-coded
CATEGORICAL_COLS = [
    "CERTLEVP", "PGMCIPAP", "PGM_P034", "PGM_P036", "PGM_P401",
    "STULOANS", "CTZSHIPP", "VISBMINP", "DDIS_FL", "GENDER2",
    "PAR1GRD", "PAR2GRD", "BEF_P140",
    "PGM_280A", "PGM_280B", "PGM_280C", "PGM_280F",
]

# Ordinal-coded columns (we’ll add safe numeric transforms + binaries)
ORDINAL_COLS = [
    "HLOSGRDP", "PREVLEVP", "GRADAGEP", "DBTOTGRD", "SCHOLARP", "BEF_160"
]

def _apply_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace column-specific missing/skip codes with NaN (conservative)."""
    out = df.copy()

    for col in out.columns:
        if col not in out.columns:
            continue
        if col in MISSING_CODES:
            codes = MISSING_CODES[col]
        else:
            codes = MISSING_CODES["default"]

        # Only apply to numeric-like columns; if strings exist, skip
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out.loc[out[col].isin(codes), col] = np.nan

    return out

def _add_missing_indicators(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}__is_missing"] = out[c].isna().astype(int)
    return out

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for NGS underemployment dataset.

    Goals:
    - Do NOT incorrectly treat legitimate category codes as missing (e.g., HLOSGRDP=6 is a valid "Master's")
    - Convert only explicit missing/skip codes to NaN (conservative mapping)
    - Add robust missingness flags + simple ordinal/binary/interaction features
    """
    out = df.copy()

    # Standardize known missing/skip codes -> NaN (conservative)
    out = _apply_missing_codes(out)

    # Ensure ordinals are numeric (with NaN allowed)
    out = _coerce_numeric(out, ORDINAL_COLS + ["CERTLEVP"])

    # Missing indicators (very useful for survey data)
    out = _add_missing_indicators(out, ORDINAL_COLS + CATEGORICAL_COLS)

    # -----------------------------------------------------------------
    # Simple high-signal engineered features (safe and interpretable)
    # -----------------------------------------------------------------

    # Scholarship / debt binaries from ordinal bins (after missing -> NaN)
    if "SCHOLARP" in out.columns:
        out["has_scholarship"] = (out["SCHOLARP"].fillna(-1) > 0).astype(int)
    if "DBTOTGRD" in out.columns:
        out["has_nongov_debt"] = (out["DBTOTGRD"].fillna(-1) > 0).astype(int)

    # Work experience bands + log transform
    if "BEF_160" in out.columns:
        # cap extreme values to avoid outliers dominating
        out["BEF_160_capped"] = out["BEF_160"].clip(lower=0, upper=97)
        out["log_BEF_160"] = np.log1p(out["BEF_160_capped"].fillna(0))

        out["has_work_exp"] = (out["BEF_160_capped"].fillna(0) > 0).astype(int)
        out["work_exp_high"] = (out["BEF_160_capped"].fillna(0) >= 24).astype(int)  # 2+ years

    # Education progression proxy: graduation education - previous education
    # (both are ordinal-coded; if either missing -> NaN)
    if "HLOSGRDP" in out.columns and "PREVLEVP" in out.columns:
        out["edu_progress"] = out["HLOSGRDP"] - out["PREVLEVP"]
        out["edu_progress_abs"] = out["edu_progress"].abs()

    # Credential level interactions (CatBoost can handle as categorical)
    # Create combined categorical tokens (string)
    def _cat_pair(a: str, b: str, name: str):
        if a in out.columns and b in out.columns:
            out[name] = (
                out[a].astype("string").fillna("__MISSING__")
                + "_x_"
                + out[b].astype("string").fillna("__MISSING__")
            )

    _cat_pair("CERTLEVP", "PGMCIPAP", "CERTLEVP_x_PGMCIPAP")
    _cat_pair("CERTLEVP", "GRADAGEP", "CERTLEVP_x_GRADAGEP")
    _cat_pair("CERTLEVP", "GENDER2", "CERTLEVP_x_GENDER2")

    # Online learning + loans interaction (binary-ish signal)
    if "PGM_P401" in out.columns and "STULOANS" in out.columns:
        cond = (
            (out["PGM_P401"].astype("string").fillna("__MISSING__") == "1")
            & (out["STULOANS"].astype("string").fillna("__MISSING__") == "1")
        )
        out["online_and_loans"] = cond.fillna(False).astype(int)

    # Ensure core categorical columns are strings for CatBoost stability
    for c in CATEGORICAL_COLS + ["CERTLEVP_x_PGMCIPAP", "CERTLEVP_x_GRADAGEP", "CERTLEVP_x_GENDER2"]:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna("__MISSING__")

    return out

def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (categorical_columns, numeric_columns) for CatBoost.
    Categorical = columns in CATEGORICAL_COLS plus any object/string/category dtype.
    """
    cols = df.columns.tolist()

    # Start with declared categorical columns if present
    cat = [c for c in CATEGORICAL_COLS if c in cols]

    # Add any string-like columns (including engineered pair features)
    for c in cols:
        if c not in cat and (
            pd.api.types.is_object_dtype(df[c]) or
            pd.api.types.is_string_dtype(df[c]) or
            pd.api.types.is_categorical_dtype(df[c])
        ):
            cat.append(c)

    num = [c for c in cols if c not in cat]
    return cat, num

