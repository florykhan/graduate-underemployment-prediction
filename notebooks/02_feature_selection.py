import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------
# Config
# ---------------------------------------------------
USE_PREPROCESS = False  # set True when src/preprocess.py exists
TRAIN_PATH = "data/ng-hackml-2026/train.csv"

TARGET = "overqualified"
IDCOL = "id"

# ---------------------------------------------------
# Imports from your pipeline
# ---------------------------------------------------
from src.features import add_features, get_feature_types

# ---------------------------------------------------
# Load + features
# ---------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
df = train.copy()

if USE_PREPROCESS:
    from src.preprocess import clean
    df = clean(df)

df = add_features(df)

y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

# id should never be useful
if IDCOL in X.columns:
    X = X.drop(columns=[IDCOL])

# CatBoost cat feature indices
feature_cols = X.columns.tolist()

# Robust: anything non-numeric is categorical (e.g. "Female")
cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]
cat_idx = [feature_cols.index(c) for c in cat_cols]

print("X:", X.shape)
print("Detected categorical columns:", len(cat_cols))
print("Example categorical cols:", cat_cols[:10])

# ---------------------------------------------------
# Holdout split
# ---------------------------------------------------
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Fix: CatBoost cannot accept NaN inside categorical columns
for c in cat_cols:
    X_tr[c] = X_tr[c].fillna("__MISSING__").astype(str)
    X_va[c] = X_va[c].fillna("__MISSING__").astype(str)

tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
va_pool = Pool(X_va, y_va, cat_features=cat_idx)

# ---------------------------------------------------
# Train "selection" model (not final tuning)
# ---------------------------------------------------
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=6,
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=200,
    od_type="Iter",
    od_wait=200
)

model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

va_pred = model.predict_proba(va_pool)[:, 1]
baseline_auc = roc_auc_score(y_va, va_pred)
print("\nBaseline AUC:", baseline_auc)
print("Best iter:", model.get_best_iteration())

# ---------------------------------------------------
# Built-in importance
# ---------------------------------------------------
fi_vals = model.get_feature_importance(type="FeatureImportance", data=va_pool)
fi = pd.DataFrame({"feature": feature_cols, "catboost_importance": fi_vals})
fi = fi.sort_values("catboost_importance", ascending=False)

print("\nTop 20 by CatBoost importance:")
print(fi.head(20).to_string(index=False))

# ---------------------------------------------------
# Permutation importance (AUC drop)
# ---------------------------------------------------
rng = np.random.default_rng(42)

def auc_on(Xdf):
    p = model.predict_proba(Pool(Xdf, cat_features=cat_idx))[:, 1]
    return roc_auc_score(y_va, p)

perm_drop = []
for col in feature_cols:
    Xp = X_va.copy()
    Xp[col] = rng.permutation(Xp[col].values)
    perm_auc = auc_on(Xp)
    perm_drop.append(baseline_auc - perm_auc)

pi = pd.DataFrame({"feature": feature_cols, "perm_auc_drop": perm_drop})
pi = pi.sort_values("perm_auc_drop", ascending=False)

print("\nTop 20 by permutation AUC drop:")
print(pi.head(20).to_string(index=False))

# ---------------------------------------------------
# Merge + choose drop list
# ---------------------------------------------------
merged = fi.merge(pi, on="feature", how="inner").sort_values("perm_auc_drop", ascending=False)

# Conservative thresholds (adjust if you want more aggressive dropping)
DROP = merged[
    (merged["perm_auc_drop"] < 0.0005) & (merged["catboost_importance"] < 1.0)
]["feature"].tolist()

print("\nDROP count:", len(DROP))
print("First 50 drop candidates:", DROP[:50])

# ---------------------------------------------------
# Sanity check: retrain without DROP
# ---------------------------------------------------
X_tr2 = X_tr.drop(columns=DROP)
X_va2 = X_va.drop(columns=DROP)

feature_cols2 = X_tr2.columns.tolist()
cat_idx2 = [feature_cols2.index(c) for c in cat_cols if c in feature_cols2]

tr_pool2 = Pool(X_tr2, y_tr, cat_features=cat_idx2)
va_pool2 = Pool(X_va2, y_va, cat_features=cat_idx2)

model2 = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=model.get_best_iteration() or 800,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=6,
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=False
)

model2.fit(tr_pool2, eval_set=va_pool2, use_best_model=True)
auc2 = roc_auc_score(y_va, model2.predict_proba(va_pool2)[:, 1])

print("\nReduced AUC:", auc2, "| Delta:", auc2 - baseline_auc)

# ---------------------------------------------------
# Save report + print paste-ready list
# ---------------------------------------------------
merged.to_csv("submissions/feature_importance_report.csv", index=False)
print("\nSaved: submissions/feature_importance_report.csv")

print("\nPaste into src/features.py:\n")
print("DROP_COLS_DEFAULT = [")
for c in DROP:
    print(f'    \"{c}\",')
print("]")

