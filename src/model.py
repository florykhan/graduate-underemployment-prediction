"""
Model module (owned by Model/Cross-validation owner).
Defines models (e.g., Logistic Regression, CatBoost).
Exposes: build_model(...) -> model instance
"""
# TODO: Integrate teammate's model.build_model() when merged
# Placeholder: simple LogisticRegression for pipeline to run

from sklearn.linear_model import LogisticRegression


def build_model(random_state: int = 42, **kwargs):
    """Build and return a model instance."""
    return LogisticRegression(random_state=random_state, max_iter=1000, **kwargs)
