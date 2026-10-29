"""Scorecard: optimal binning, WoE, IV, logistic regression, points and grades.

Conventions (optbinning): WoE = ln(share of goods / share of bads) in a bin, so a higher WoE
means a safer bin, and every logistic coefficient on WoE must be negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression

from credit_risk.config import load_config

SC = load_config()["scorecard"]
# Scaling: 600 points at odds 50:1, +20 points (PDO) each time the odds double.
FACTOR = SC["pdo"] / np.log(2)
OFFSET = SC["base_points"] - FACTOR * np.log(SC["base_odds"])


# ---------- binning, WoE, IV, correlation filter ----------


def fit_binning(X: pd.DataFrame, y: np.ndarray) -> BinningProcess:
    """Optimal binning of every column of X, fitted on the training data only.

    Numeric variables get a monotonic WoE trend (ascending or descending, chosen per variable),
    every bin holds at least `min_bin_size` of the rows, missing values get their own bin,
    and variables with IV below `min_iv` are marked as not selected.
    """
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric_params = {"monotonic_trend": "auto_asc_desc"}
    # binning_fit_params holds per-variable settings: the monotonic constraint, numeric only
    return BinningProcess(
        variable_names=list(X.columns),
        categorical_variables=categorical,
        min_bin_size=SC["min_bin_size"],
        binning_fit_params={
            col_num: numeric_params for col_num in X.columns if col_num not in categorical
        },
        selection_criteria={"iv": {"min": SC["min_iv"]}},
    ).fit(X, y)


def woe_transform(binning: BinningProcess, X: pd.DataFrame) -> pd.DataFrame:
    """Replace every value by the WoE of its bin, INCLUDING the missing bin.

    optbinning's transform gives missing values a WoE of 0 by default (an "average" risk);
    "empirical" uses the WoE learned for the missing bin, as decided in the EDA.
    """
    return binning.transform(
        X, metric="woe", metric_missing="empirical", metric_special="empirical"
    )


def iv_table(binning: BinningProcess) -> pd.DataFrame:
    """Name, IV, selected flag and number of bins per variable, strongest first."""
    binning_summary = binning.summary()[["name", "iv", "selected", "n_bins"]].copy()
    binning_summary["iv"] = binning_summary["iv"].astype("float")
    return binning_summary.sort_values("iv", ascending=False).reset_index(drop=True)


def drop_correlated(woe: pd.DataFrame, iv: pd.Series, threshold: float = 0.8) -> list[str]:
    """Greedy filter: walk the variables from the highest IV down, keep one only if its WoE
    correlation with every variable already kept is at most `threshold`."""
    # Why greedy by IV: of two near duplicates, we keep the more predictive one.
    corr_matrix = woe.corr().abs()
    kept = []
    for col in iv.loc[woe.columns].sort_values(ascending=False).index:
        correlated = any(corr_matrix.loc[col, k] > threshold for k in kept)
        if not correlated:
            kept.append(col)
    return kept


# ---------- logistic regression, points, grades ----------


def fit_logistic_with_sign_check(
    woe: pd.DataFrame, y: np.ndarray, features: list[str]
) -> tuple[LogisticRegression, list[str]]:
    """Logistic regression on WoE; while a coefficient is positive, drop that variable and refit.

    A positive coefficient means a safer bin raises the PD: impossible to explain, usually a
    side effect of collinearity. `features` must be ordered by decreasing IV: the weakest
    wrong-sign variable is dropped first.
    """
    features = list(features)  # a copy: never modify the caller's list
    while True:
        model = LogisticRegression(max_iter=1000).fit(woe[features], y)
        wrong = [
            wrong_f for wrong_f, coef in zip(features, model.coef_[0], strict=True) if coef >= 0
        ]
        if not wrong:
            return model, features
        features.remove(wrong[-1])


def pd_to_score(pd_hat: np.ndarray) -> np.ndarray:
    """Points = OFFSET + FACTOR * ln(odds of being good): 600 at odds 50:1, +20 when odds double."""
    pd_hat = np.asarray(pd_hat, dtype=float)
    odd_goods = (1 - pd_hat) / pd_hat
    return OFFSET + FACTOR * np.log(odd_goods)


def fit_grade_cutoffs(scores: np.ndarray) -> np.ndarray:
    """Four score cut-offs = quintiles of the TRAINING scores (each grade holds 20 % of train)."""
    return np.quantile(scores, [0.2, 0.4, 0.6, 0.8])


def assign_grades(scores: np.ndarray, cutoffs: np.ndarray) -> pd.Series:
    """Grade A for the highest scores (lowest risk) down to E. Works on any score."""
    labels = SC["grades"][::-1]  # E for the lowest band, A for the highest
    # np.searchsorted gives, for each score, how many cut-offs are below it: 0 (E) to 4 (A).
    return pd.Series(np.asarray(labels)[np.searchsorted(cutoffs, scores, side="right")])


def grade_table(grades: pd.Series, y: np.ndarray) -> pd.DataFrame:
    """Count and observed default rate per grade, A first."""
    # np.asarray drops the index: y comes from a shuffled split, grades from a 0..n index
    df = pd.DataFrame({"grade": np.asarray(grades), "TARGET": np.asarray(y)})
    return df.groupby("grade")["TARGET"].agg(count="size", default_rate="mean")


@dataclass
class ScorecardModel:
    """Everything needed to score a new applicant: bins, selected variables, logistic, cut-offs."""

    binning: BinningProcess
    features: list[str]
    logistic: LogisticRegression
    cutoffs: np.ndarray

    def woe(self, X: pd.DataFrame) -> pd.DataFrame:
        return woe_transform(self.binning, X)[self.features]

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.logistic.predict_proba(self.woe(X))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return pd_to_score(self.predict_pd(X))

    def grade(self, X: pd.DataFrame) -> pd.Series:
        return assign_grades(self.score(X), self.cutoffs)

    def points(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points per variable; they add up to the score. Base for the reason codes.

        score = OFFSET - FACTOR * (b0 + sum b_i * woe_i): each variable earns -FACTOR * b_i * woe_i
        and the constant part is shared equally between the variables.
        """
        b0, b = self.logistic.intercept_[0], self.logistic.coef_[0]
        base = (OFFSET - FACTOR * b0) / len(self.features)  # constant part, shared equally
        return base - FACTOR * self.woe(X) * b  # one column per variable


def fit_scorecard(
    X: pd.DataFrame, y: np.ndarray, max_features: int | None = None
) -> ScorecardModel:
    """Full scorecard on the training data: bins -> IV filter -> correlation filter ->
    sign check -> keep the `max_features` strongest -> grade cut-offs."""
    max_features = max_features or SC["max_features"]
    binning = fit_binning(X, y)
    ivs = iv_table(binning)
    iv = ivs.set_index("name")["iv"]
    selected = ivs.loc[ivs["selected"], "name"].tolist()
    woe = woe_transform(binning, X)[selected]

    candidates = drop_correlated(woe, iv, threshold=SC["max_corr"])
    _, candidates = fit_logistic_with_sign_check(woe, y, candidates)
    features = candidates[:max_features]
    # Refit on the final features: dropping variables can flip a sign, so check again.
    # Grade cut-offs come from the TRAINING scores: they are part of the model, like the bins.
    logistic, features = fit_logistic_with_sign_check(woe, y, features)
    model = ScorecardModel(binning, features, logistic, cutoffs=np.array([]))
    model.cutoffs = fit_grade_cutoffs(model.score(X))
    return model
