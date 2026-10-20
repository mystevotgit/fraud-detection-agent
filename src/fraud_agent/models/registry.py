from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from fraud_agent.features.preprocessing import build_scaled_pipeline


def build_supervised_models(random_seed: int = 582) -> dict[str, object]:
    """Return course-aligned laptop-friendly candidate models."""
    return {
        "logistic_regression": build_scaled_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_seed),
            scale_pca_features=False,
        ),
        "decision_tree": build_scaled_pipeline(
            DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=25,
                class_weight="balanced",
                random_state=random_seed,
            ),
            scale_pca_features=False,
        ),
        "random_forest": build_scaled_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_seed,
            ),
            scale_pca_features=False,
        ),
        "linear_svm_calibrated": build_scaled_pipeline(
            CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", random_state=random_seed, max_iter=5000),
                method="sigmoid",
                cv=3,
            ),
            scale_pca_features=True,
        ),
        "mlp": build_scaled_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64,),
                alpha=0.001,
                early_stopping=True,
                validation_fraction=0.15,
                max_iter=120,
                random_state=random_seed,
            ),
            scale_pca_features=True,
        ),
    }


def build_unsupervised_models(random_seed: int = 582) -> dict[str, object]:
    return {
        "isolation_forest": build_scaled_pipeline(
            IsolationForest(
                contamination=0.002,
                n_estimators=150,
                random_state=random_seed,
                n_jobs=-1,
            ),
            scale_pca_features=True,
        )
    }

