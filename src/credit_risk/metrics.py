"""Credit-risk metrics: Gini and KS are what a bank reports."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def gini(y: np.ndarray, p: np.ndarray) -> float:
    """Gini = 2 * AUC - 1: 0 for a random model, 1 for a perfect ranking of bads above goods."""
    return float(2 * roc_auc_score(y, p) - 1)


def ks(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: the largest gap between the cumulative distributions of bads and goods.

    Along the ROC curve, TPR is the share of bads caught and FPR the share of goods caught
    at the same threshold, so KS = max(TPR - FPR).
    """
    # KS is the classic scorecard separation measure, read directly on the score scale.
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Metrics reported for every model: ranking (AUC, Gini, KS, PR-AUC), calibration (Brier)."""
    return {
        "auc": float(roc_auc_score(y, p)),
        "gini": float(gini(y, p)),
        "ks": float(ks(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
