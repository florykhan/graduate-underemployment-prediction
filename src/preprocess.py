# Import required libraries
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

# =============================================================================
# SPECIAL VALUE ENCODING FOR CATBOOST
# =============================================================================
# For CATEGORICAL features: Use string sentinels (CatBoost handles them well)
# For NUMERIC features: Use NaN (CatBoost has native NaN handling with special splits)
#
# Why this approach?
# - CatBoost natively handles NaN in numeric features with optimal split finding
# - String sentinels for categorical clearly separate "missing" from "skip"
# - Avoids corrupting numeric relationships (e.g., -1 debt doesn't make sense)

MISSING_VALUE_CAT = "MISSING"      # Not stated / unknown (for categorical)
VALID_SKIP_VALUE_CAT = "SKIP"      # Valid skip (for categorical)
MISSING_VALUE_NUM = np.nan         # For numeric features - CatBoost handles natively

# Define feature types for CatBoost
# Categorical features: CatBoost will treat these as discrete categories
CATEGORICAL_FEATURES = [
    'CERTLEVP',      # Credential level (ordinal but treated as categorical)
    'HLOSGRDP',      # Highest level of schooling
    'PREVLEVP',      # Previous education level
    'GENDER2',       # Gender
    'VISBMINP',      # Visible minority
    'IMMIGRP',       # Immigration status
    'DDIS_FL',       # Disability flag
    'PGM_P034',      # Program type
    'PGM_P401',      # Another program attribute
    'PGM_P036',      # Program attribute
    'PGM_ONLN',      # Online program
    'PAR1EDUC',      # Parent 1 education
    'PAR2EDUC',      # Parent 2 education
    'BEF_P140',      # Activity before program
]

# Numeric/Ordinal features: These have meaningful numeric relationships
NUMERIC_FEATURES = [
    'DBTOTGRD',      # Total debt at graduation (continuous)
    'SCHOLARP',      # Scholarship amount (continuous)
    'GRADAGEP',      # Age at graduation (continuous)
    'BEF_160',       # Work experience months (continuous)
    'PGMCIPAP',      # Field of study code (high cardinality, treat as numeric)
]


def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Clean the dataset by:
    1. Converting mixed-type columns to consistent numeric
    2. Handling special codes (6, 9, 99) with meaningful sentinel values
    3. Creating clean versions of features
    
    Special Code Handling:
    - Code 6 (Valid Skip): Set to -2 (question not applicable to respondent)
    - Code 9 (Not Stated): Set to -1 (respondent chose not to answer)
    - Code 99 (Not Stated for continuous): Set to -1
    - NaN: Set to -1
    
    Args:
        df: Input dataframe
        is_train: Whether this is training data (for logging)
    
    Returns:
        Cleaned dataframe with _clean suffix columns
    """
    df_clean = df.copy()
    
    # String to numeric mappings for mixed-type columns
    string_mappings = {
        'GENDER2': {'Male': 1.0, 'Female': 2.0},
        'VISBMINP': {'Yes': 1.0, 'No': 2.0},
        'DDIS_FL': {'With disability': 1.0, 'Without disability': 2.0}
    }
    
    # Apply string mappings
    for col, mapping in string_mappings.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(mapping)
    
    # Get feature columns (exclude id and target)
    feature_cols = [col for col in df_clean.columns if col not in ['id', 'overqualified']]
    
    # Convert all feature columns to numeric
    for col in feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Create cleaned versions with special code handling
    for col in feature_cols:
        clean_col_name = f"{col}_clean"
        df_clean[clean_col_name] = df_clean[col].copy()
        
        # Handle based on feature type
        if col in NUMERIC_FEATURES:
            # Numeric features: use NaN (CatBoost handles natively)
            # 99 typically means "not stated" for continuous vars
            df_clean.loc[df_clean[clean_col_name] == 99, clean_col_name] = MISSING_VALUE_NUM
            df_clean.loc[df_clean[clean_col_name] == 6, clean_col_name] = MISSING_VALUE_NUM  # Skip also becomes NaN
            # Keep existing NaN as NaN
        else:
            # Categorical features: use string sentinels
            # First convert to string to allow string sentinels
            df_clean[clean_col_name] = df_clean[clean_col_name].astype(str)
            
            # Replace special codes with meaningful strings
            df_clean.loc[df_clean[col] == 9, clean_col_name] = MISSING_VALUE_CAT
            df_clean.loc[df_clean[col] == 6, clean_col_name] = VALID_SKIP_VALUE_CAT
            df_clean.loc[df_clean[col].isna(), clean_col_name] = MISSING_VALUE_CAT
            
            # Clean up "nan" strings that might have been created
            df_clean.loc[df_clean[clean_col_name] == 'nan', clean_col_name] = MISSING_VALUE_CAT
    
    return df_clean


def get_catboost_feature_indices(feature_names: List[str]) -> List[int]:
    """
    Get indices of categorical features for CatBoost's cat_features parameter.
    
    CatBoost handles categorical features specially - it uses ordered target 
    statistics instead of one-hot encoding, which is more efficient and often
    more effective.
    
    Args:
        feature_names: List of feature column names
    
    Returns:
        List of indices for categorical features
    """
    cat_indices = []
    for idx, name in enumerate(feature_names):
        if name in CATEGORICAL_FEATURES:
            cat_indices.append(idx)
    return cat_indices


def prepare_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for CatBoost by ensuring correct data types.
    
    For CatBoost:
    - Categorical features: keep as string (CatBoost handles string categories)
    - Numeric features: keep as float with NaN (CatBoost handles NaN natively)
    
    Args:
        df: Cleaned dataframe
    
    Returns:
        Dataframe ready for CatBoost
    """
    df_cb = df.copy()
    
    for col in df_cb.columns:
        if col in CATEGORICAL_FEATURES:
            # Keep as string for categorical (CatBoost handles strings well)
            df_cb[col] = df_cb[col].astype(str)
        elif col in NUMERIC_FEATURES:
            # Keep as float for numeric features (NaN preserved)
            df_cb[col] = pd.to_numeric(df_cb[col], errors='coerce')
        else:
            # Default: try numeric first, then string
            try:
                df_cb[col] = pd.to_numeric(df_cb[col], errors='raise')
            except (ValueError, TypeError):
                df_cb[col] = df_cb[col].astype(str)
    
    return df_cb


def preprocess_dataset(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Complete preprocessing pipeline for train and test datasets.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        verbose: Whether to print progress
    
    Returns:
        Tuple of (train_features, test_features, categorical_indices)
    """
    # Clean both datasets
    train_clean = clean_data(train_df, is_train=True)
    test_clean = clean_data(test_df, is_train=False)
    
    if verbose:
        print(f"Training set shape after cleaning: {train_clean.shape}")
        print(f"Test set shape after cleaning: {test_clean.shape}")
    
    # Extract clean columns only
    clean_cols = [col for col in train_clean.columns if col.endswith('_clean')]
    train_final = train_clean[clean_cols].copy()
    test_final = test_clean[clean_cols].copy()
    
    # Remove _clean suffix from column names
    train_final.columns = [col.replace('_clean', '') for col in clean_cols]
    test_final.columns = [col.replace('_clean', '') for col in clean_cols]
    
    # Prepare for CatBoost (correct data types)
    train_final = prepare_for_catboost(train_final)
    test_final = prepare_for_catboost(test_final)
    
    # Get categorical feature indices for CatBoost
    cat_indices = get_catboost_feature_indices(train_final.columns.tolist())
    
    if verbose:
        print(f"\nFinal training shape: {train_final.shape}")
        print(f"Final test shape: {test_final.shape}")
        print(f"Categorical features ({len(cat_indices)}): {[train_final.columns[i] for i in cat_indices]}")
        print(f"Numeric features ({len(train_final.columns) - len(cat_indices)}): "
              f"{[c for c in train_final.columns if c not in CATEGORICAL_FEATURES]}")
        
        # Report sentinel value usage
        print("\nMissing value summary:")
        for col in train_final.columns:
            if col in CATEGORICAL_FEATURES:
                # Categorical: count string sentinels
                missing_count = (train_final[col] == MISSING_VALUE_CAT).sum()
                skip_count = (train_final[col] == VALID_SKIP_VALUE_CAT).sum()
                if missing_count > 0 or skip_count > 0:
                    print(f"  {col}: {missing_count} MISSING, {skip_count} SKIP")
            else:
                # Numeric: count NaN
                nan_count = train_final[col].isna().sum()
                if nan_count > 0:
                    print(f"  {col}: {nan_count} NaN (handled natively by CatBoost)")
    
    return train_final, test_final, cat_indices

