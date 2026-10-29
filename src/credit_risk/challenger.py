"""Challenger: gradient boosting (XGBoost) on the same features as the scorecard.

Gradient boosting adds small trees one after another; each new tree fits the errors (the
gradient of the log-loss) left by the trees before it, scaled down by the learning rate.
Depth, min_child_weight, lambda and gamma limit how much each tree can learn; subsampling
rows and columns adds randomness against overfitting. scale_pos_weight makes every bad
count about 11 times more than a good: the ranking improves, the probabilities are inflated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

SEED = 42
BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",  # bins every feature into 256 buckets: fast on 200k rows
    "enable_categorical": True,  # native handling of the text columns
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
}

# A small, justified search: depth (interactions), learning rate (step size), and the two
# regularisations that matter most on noisy credit data.
PARAM_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 50, "reg_lambda": 5, "gamma": 0},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 50, "reg_lambda": 5, "gamma": 1},
    {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 100, "reg_lambda": 10, "gamma": 1},
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 200, "reg_lambda": 10, "gamma": 2},
]


def prepare(
    X: pd.DataFrame, categories: dict[str, list[str]] | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Text columns become pandas categories with a FIXED list of levels learned on train.

    A level never seen in training becomes missing: XGBoost then sends it the default way.
    """
    X = X.copy()
    text = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if categories is None:
        categories = {c: sorted(X[c].dropna().unique().tolist()) for c in text}
    for col, levels in categories.items():
        known = X[col].where(X[col].isin(levels))  # unseen level -> NaN, explicitly
        X[col] = pd.Categorical(known, categories=levels)
    return X, categories


def _classifier(params: dict, n_rounds: int, y: np.ndarray, early_stopping: int | None = None):
    scale = float((y == 0).sum() / max((y == 1).sum(), 1))  # goods / bads, about 11.4 here
    return XGBClassifier(
        **BASE_PARAMS,
        **params,
        n_estimators=n_rounds,
        scale_pos_weight=scale,
        early_stopping_rounds=early_stopping,
    )


def cv_search(
    X: pd.DataFrame,
    y: np.ndarray,
    grid: list[dict] = PARAM_GRID,
    n_splits: int = 5,
    max_rounds: int = 2_000,
    patience: int = 50,
) -> pd.DataFrame:
    """Stratified k-fold CV on the TRAINING set only, one row per parameter combination.

    In each fold, the held-out part is the early-stopping set: trees are added until the AUC
    on it has not improved for `patience` rounds. best_rounds is the average stopping point.
    """
    y = np.asarray(y)
    X, categories = prepare(X)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for params in grid:
        aucs, rounds = [], []
        for fit_idx, val_idx in folds.split(X, y):
            model = _classifier(params, max_rounds, y[fit_idx], early_stopping=patience)
            model.fit(
                X.iloc[fit_idx], y[fit_idx], eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False
            )
            aucs.append(roc_auc_score(y[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1]))
            rounds.append(model.best_iteration + 1)
        rows.append(
            {
                **params,
                "cv_auc": float(np.mean(aucs)),
                "cv_auc_std": float(np.std(aucs)),
                "best_rounds": int(np.mean(rounds)),
            }
        )
    table = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    return table.reset_index(drop=True)


@dataclass
class ChallengerModel:
    """Fitted XGBoost plus the category levels learned on train."""

    model: XGBClassifier
    categories: dict[str, list[str]]

    def prepared(self, X: pd.DataFrame) -> pd.DataFrame:
        return prepare(X, self.categories)[0]

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """RAW probabilities: inflated by scale_pos_weight, calibrate them before any decision."""
        return self.model.predict_proba(self.prepared(X))[:, 1]


def fit_challenger(X: pd.DataFrame, y: np.ndarray, params: dict, n_rounds: int) -> ChallengerModel:
    """Final fit on the whole training set with the number of trees chosen by CV."""
    y = np.asarray(y)
    prepared, categories = prepare(X)
    model = _classifier(params, n_rounds, y).fit(prepared, y, verbose=False)
    return ChallengerModel(model, categories)
