"""Tests for reason codes, SHAP contributions and the fairness table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.challenger import fit_challenger
from credit_risk.explain import (
    REASON_TEXT,
    best_points,
    fairness_table,
    reason_codes,
    shap_contributions,
)
from credit_risk.scorecard import fit_scorecard


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 4_000
    ext = rng.uniform(0, 1, n)
    late = rng.uniform(0, 0.5, n)
    job = rng.choice(["Accountants", "Drivers", "Laborers"], n)
    logit = -1.5 - 3 * ext + 3 * late + np.where(job == "Drivers", 0.5, 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"EXT_SOURCE_3": ext, "LATE_SHARE": late, "OCCUPATION_TYPE": job})
    return X, y


@pytest.fixture(scope="module")
def scorecard(data):
    X, y = data
    return fit_scorecard(X, y, max_features=3)


def test_best_points_is_never_below_an_applicants_points(data, scorecard) -> None:
    X, _ = data
    best = best_points(scorecard)
    points = scorecard.points(X)
    assert set(best.index) == set(scorecard.features)
    assert (points.le(best + 1e-9)).all().all()


def test_reason_codes_are_the_three_biggest_point_losses(data, scorecard) -> None:
    X, _ = data
    worst = X.assign(score=scorecard.score(X)).sort_values("score").head(1).drop(columns="score")
    reasons = reason_codes(scorecard, worst, n=3)[0]
    assert len(reasons) == 3
    lost = [r["points_lost"] for r in reasons]
    assert lost == sorted(lost, reverse=True) and lost[0] > 0
    assert all(r["text"] for r in reasons)


def test_every_scorecard_variable_has_a_plain_language_reason() -> None:
    from credit_risk.config import ROOT

    model_file = ROOT / "models" / "scorecard.joblib"
    if not model_file.exists():
        pytest.skip("no trained scorecard (run make train)")
    import joblib

    missing = set(joblib.load(model_file).features) - set(REASON_TEXT)
    assert not missing, f"add a plain-language reason for {missing}"


def test_shap_contributions_add_up_to_the_model_output(data) -> None:
    X, y = data
    model = fit_challenger(X, y, {"max_depth": 3, "learning_rate": 0.1}, n_rounds=50)
    contrib = shap_contributions(model, X.head(50))
    assert list(contrib.columns) == [*X.columns, "BIAS"]
    logit = np.log(model.predict_pd(X.head(50)) / (1 - model.predict_pd(X.head(50))))
    np.testing.assert_allclose(contrib.sum(axis=1), logit, rtol=1e-4, atol=1e-4)


def test_fairness_table() -> None:
    group = pd.Series(["F", "F", "M", "M", "M"])
    approved = np.array([True, False, True, True, False])
    y = np.array([0, 1, 1, 0, 1])
    table = fairness_table(group, approved, y)
    assert table.loc["F", "approval_rate"] == 0.5
    assert table.loc["M", "default_rate_approved"] == 0.5
    # ratio to the most approved group ("four-fifths rule": below 0.8 deserves a look)
    assert table.loc["M", "approval_ratio"] == pytest.approx(1.0)
    assert table.loc["F", "approval_ratio"] == pytest.approx(0.5 / (2 / 3))
