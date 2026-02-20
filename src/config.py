"""
Configuration for the ML hackathon pipeline.
Paths, model settings, and constantsx
"""
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths (raw CSVs are read-only)
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Output paths
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODEL_ARTIFACT_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Target and identifier columns
TARGET_COL = "overqualified"
ID_COL = "id"

# Validation settings
VAL_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 5
