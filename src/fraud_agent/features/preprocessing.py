from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_SCALE_FEATURES = ["Time", "Amount"]
FEATURE_COLUMNS = ["Time", *PCA_FEATURES, "Amount"]


def build_preprocessor(scale_pca_features: bool = False) -> ColumnTransformer:
    """Build a leakage-safe sklearn preprocessor to be fit inside training pipelines."""
    passthrough_features = [] if scale_pca_features else PCA_FEATURES
    scale_features = FEATURE_COLUMNS if scale_pca_features else RAW_SCALE_FEATURES

    return ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(), scale_features),
            ("pca_passthrough", "passthrough", passthrough_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_scaled_pipeline(estimator, scale_pca_features: bool = False) -> Pipeline:
    """Attach preprocessing to an estimator so transformations see training folds only."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_pca_features=scale_pca_features)),
            ("model", estimator),
        ]
    )

