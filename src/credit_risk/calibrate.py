"""Probability calibration: a PD of 10 % must mean that 10 % of such clients default.

Fitted on the CALIBRATION set (never on train, where the model is overconfident, never on
test, which is reserved for the final measure). Two methods:
- sigmoid (Platt): a logistic regression on the logit of the raw score, 2 parameters, smooth;
- isotonic: a non-decreasing step function, flexible, needs more data.
Both keep the ranking (Gini unchanged): only the probability scale moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    """Maps a raw probability to a calibrated one. method: "sigmoid" or "isotonic"."""

    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError(f"unknown calibration method: {method}")
        self.method = method

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "sigmoid":
            self.model_ = LogisticRegression(C=1e6).fit(_logit(p_raw), y)  # no penalty: pure Platt
        else:
            self.model_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
            self.model_.fit(np.asarray(p_raw, dtype=float), y)
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            return self.model_.predict_proba(_logit(p_raw))[:, 1]
        return self.model_.predict(np.asarray(p_raw, dtype=float))


def oof_calibrate(
    p_raw: np.ndarray, y: np.ndarray, method: str, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    """Out-of-fold calibrated probabilities: each row is calibrated by a calibrator fitted on
    the other folds. Gives an honest Brier score on the calibration set itself."""
    p_raw, y = np.asarray(p_raw, dtype=float), np.asarray(y)
    out = np.empty_like(p_raw)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, pred_idx in folds.split(p_raw, y):
        out[pred_idx] = Calibrator(method).fit(p_raw[fit_idx], y[fit_idx]).predict(p_raw[pred_idx])
    return out


def compare_calibrations(p_raw: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Brier score and mean PD of the raw scores and of both methods (out-of-fold)."""
    rows = {"raw": np.asarray(p_raw, dtype=float)}
    for method in ("sigmoid", "isotonic"):
        rows[method] = oof_calibrate(p_raw, y, method)
    return pd.DataFrame(
        {
            "brier": {k: brier_score_loss(y, p) for k, p in rows.items()},
            "mean_pd": {k: float(p.mean()) for k, p in rows.items()},
        }
    )


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Rows sorted by predicted PD, cut into equal-size bins: mean prediction vs observed rate.
    On a calibrated model the two columns are equal (the points sit on the diagonal)."""
    frame = pd.DataFrame({"p": np.asarray(p, dtype=float), "y": np.asarray(y)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    return frame.groupby("bin").agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), count=("y", "size")
    )
