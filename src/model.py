"""
Model module: CatBoost classifier for overqualification prediction.
Exposes: build_model(...) -> CatBoost classifier instance
"""
from catboost import CatBoostClassifier


def build_model(
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 42,
    verbose: int = 100,
    early_stopping_rounds: int | None = 20,
    **kwargs,
) -> CatBoostClassifier:
    """
    Build and return a CatBoost binary classifier for overqualification prediction.
    Tuned for NGS-style tabular data with many categorical features.
    """
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        verbose=verbose,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="Accuracy",
        **kwargs,
    )
