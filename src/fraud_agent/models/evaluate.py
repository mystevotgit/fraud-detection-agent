from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    balanced_accuracy: float
    matthews_corrcoef: float
    false_positive_rate: float
    false_negative_rate: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def binary_classification_metrics(y_true, probabilities, threshold: float = 0.5) -> BinaryMetrics:
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    predictions = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return BinaryMetrics(
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        pr_auc=float(average_precision_score(labels, probs)),
        roc_auc=float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan"),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        matthews_corrcoef=float(matthews_corrcoef(labels, predictions)),
        false_positive_rate=float(fpr),
        false_negative_rate=float(fnr),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def timed_predict_proba(model, X) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
    else:
        raise TypeError("Model must provide predict_proba or decision_function.")
    elapsed = time.perf_counter() - start
    return probabilities, elapsed

