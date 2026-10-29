"""Tests for the credit metrics and a scorecard that a credit committee would accept."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from credit_risk.metrics import gini, ks, summary
from credit_risk.scorecard import (
    assign_grades,
    drop_correlated,
    fit_binning,
    fit_grade_cutoffs,
    fit_logistic_with_sign_check,
    fit_scorecard,
    grade_table,
    iv_table,
    pd_to_score,
    woe_transform,
)

# ---------- metrics ----------


def test_gini_perfect_random_and_inverted() -> None:
    y = np.array([0, 0, 1, 1])
    assert gini(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert gini(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(-1.0)
    assert gini(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_ks_is_the_max_gap_between_bad_and_good_distributions() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    assert ks(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(1.0)
    # one bad among the goods: the best cut separates 3 goods from 2 bads -> KS = 1 - 1/3
    assert ks(y, np.array([0.1, 0.2, 0.6, 0.5, 0.8, 0.9])) == pytest.approx(2 / 3)


def test_summary_contains_the_bank_metrics() -> None:
    out = summary(np.array([0, 1, 0, 1]), np.array([0.2, 0.7, 0.4, 0.6]))
    assert set(out) == {"auc", "gini", "ks", "pr_auc", "brier"}
    assert out["gini"] == pytest.approx(2 * out["auc"] - 1)


# ---------- synthetic credit data ----------


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    """5,000 applicants: one strong score, a near copy of it, pure noise, and a category."""
    rng = np.random.default_rng(42)
    n = 5_000
    score = rng.normal(size=n)
    pd_true = 1 / (1 + np.exp(2.5 + 1.2 * score))  # higher score, lower risk
    y = (rng.random(n) < pd_true).astype(int)
    job = rng.choice(["clerk", "manager", "driver"], size=n)
    y = np.where((job == "driver") & (rng.random(n) < 0.05), 1, y)
    X = pd.DataFrame(
        {
            "SCORE": score,
            "SCORE_COPY": score + rng.normal(scale=0.05, size=n),
            "NOISE": rng.normal(size=n),
            "JOB": job,
        }
    )
    return X, y


@pytest.fixture(scope="module")
def binning(data):
    X, y = data
    return fit_binning(X, y)


# ---------- binning, WoE, IV ----------


def test_iv_table_ranks_strong_variables_first_and_rejects_noise(binning) -> None:
    table = iv_table(binning)
    assert list(table.columns[:3]) == ["name", "iv", "selected"]
    assert table.iloc[0]["name"] in {"SCORE", "SCORE_COPY"}
    noise = table.set_index("name").loc["NOISE"]
    assert noise["iv"] < 0.02 and not noise["selected"]


def test_woe_is_monotonic_for_a_monotonic_variable(binning) -> None:
    woe = binning.get_binned_variable("SCORE").binning_table.build()["WoE"]
    bins = woe.iloc[:-3].astype(float)  # drop Special, Missing and Totals rows
    assert bins.is_monotonic_increasing or bins.is_monotonic_decreasing


def test_missing_values_get_the_woe_of_their_own_bin() -> None:
    """optbinning gives missing values a WoE of 0 by default: the scorecard must use the
    WoE learned for the missing bin (here missing values are riskier, so a negative WoE)."""
    rng = np.random.default_rng(1)
    n = 4_000
    x = rng.normal(size=n)
    missing = rng.random(n) < 0.2
    y = (rng.random(n) < np.where(missing, 0.30, 1 / (1 + np.exp(2.5 + x)))).astype(int)
    X = pd.DataFrame({"X": np.where(missing, np.nan, x)})
    binning = fit_binning(X, y)
    woe_missing = woe_transform(binning, X)["X"][missing]
    assert woe_missing.nunique() == 1 and woe_missing.iloc[0] < -0.5


def test_drop_correlated_keeps_the_stronger_of_a_pair(data, binning) -> None:
    X, _ = data
    iv = iv_table(binning).set_index("name")["iv"]
    woe = woe_transform(binning, X)[["SCORE", "SCORE_COPY", "JOB"]]
    kept = drop_correlated(woe, iv, threshold=0.8)
    assert len({"SCORE", "SCORE_COPY"} & set(kept)) == 1
    assert "JOB" in kept


def test_sign_check_leaves_only_negative_coefficients(data, binning) -> None:
    X, y = data
    woe = woe_transform(binning, X)
    model, kept = fit_logistic_with_sign_check(woe, y, ["SCORE", "SCORE_COPY", "JOB"])
    assert (model.coef_[0] < 0).all(), "higher WoE (safer bin) must lower the PD"
    assert len(kept) == model.coef_.shape[1]


# ---------- points and grades ----------


def test_pd_to_score_matches_the_scaling() -> None:
    assert pd_to_score(np.array([1 / 51]))[0] == pytest.approx(600)  # odds 50:1
    assert pd_to_score(np.array([1 / 101]))[0] == pytest.approx(600 + 20 * math.log2(100 / 50))
    low, high = pd_to_score(np.array([0.02, 0.20]))
    assert low > high, "a riskier applicant gets fewer points"


def test_grades_are_ordered_and_learned_on_train_only() -> None:
    train_scores = np.arange(100, dtype=float)
    cutoffs = fit_grade_cutoffs(train_scores)
    assert len(cutoffs) == 4
    grades = assign_grades(np.array([99.0, 0.0, 1_000.0, -1_000.0]), cutoffs)
    assert grades.tolist() == ["A", "E", "A", "E"]  # unseen scores still get a grade


def test_grade_table_reports_count_and_default_rate() -> None:
    table = grade_table(pd.Series(["A", "A", "B", "B"]), np.array([0, 0, 0, 1]))
    assert table.loc["A", "count"] == 2 and table.loc["B", "default_rate"] == 0.5


# ---------- full scorecard ----------


def test_fit_scorecard_end_to_end(data) -> None:
    X, y = data
    model = fit_scorecard(X, y, max_features=3)
    pd_hat = model.predict_pd(X)
    assert ((pd_hat > 0) & (pd_hat < 1)).all()
    assert len(model.features) <= 3
    assert gini(y, pd_hat) > 0.5
    scores = model.score(X)
    assert np.corrcoef(scores, pd_hat)[0, 1] < 0  # more points, less risk


def test_points_add_up_to_the_score(data) -> None:
    X, y = data
    model = fit_scorecard(X, y, max_features=3)
    points = model.points(X.head(20))
    assert list(points.columns) == model.features
    np.testing.assert_allclose(points.sum(axis=1), model.score(X.head(20)), rtol=1e-9)


# ---------- real data ----------


@pytest.mark.data
def test_real_scorecard() -> None:
    from credit_risk.config import load_config
    from credit_risk.data import split

    features = pd.read_parquet("data/processed/application_train_features.parquet")
    train, calib, _ = split(features)
    exclude = load_config()["model"]["exclude"]
    model = fit_scorecard(train.drop(columns=exclude), train["TARGET"].to_numpy())
    pd_calib = model.predict_pd(calib.drop(columns=exclude))
    assert len(model.features) == 20
    assert "CODE_GENDER" not in model.features
    assert round(gini(calib["TARGET"], pd_calib), 3) == 0.494
    rates = grade_table(model.grade(calib.drop(columns=exclude)), calib["TARGET"])["default_rate"]
    assert rates.is_monotonic_increasing, "grade A must default less than B, B less than C..."
