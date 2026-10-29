"""Tests for the XGBoost challenger and the probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.calibrate import (
    Calibrator,
    compare_calibrations,
    oof_calibrate,
    reliability_table,
)
from credit_risk.challenger import ChallengerModel, cv_search, fit_challenger, prepare
from credit_risk.metrics import gini


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(3)
    n = 3_000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    job = rng.choice(["clerk", "driver", "manager"], size=n)
    logit = -2.5 - 1.2 * x1 + 0.8 * x2 * (x2 > 0) + np.where(job == "driver", 0.7, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"X1": x1, "X2": np.where(rng.random(n) < 0.1, np.nan, x2), "JOB": job})
    return X, y


# ---------- challenger ----------


def test_prepare_turns_text_into_fixed_categories(data) -> None:
    X, _ = data
    prepared, categories = prepare(X)
    assert str(prepared["JOB"].dtype) == "category"
    new = pd.DataFrame({"X1": [0.0], "X2": [0.0], "JOB": ["pilot"]})  # unseen category
    again, _ = prepare(new, categories)
    assert list(again["JOB"].cat.categories) == list(prepared["JOB"].cat.categories)
    assert again["JOB"].isna().iloc[0], "an unseen category becomes missing, not a new code"


def test_cv_search_uses_early_stopping_and_reports_each_combination(data) -> None:
    X, y = data
    grid = [{"max_depth": 2, "learning_rate": 0.1}, {"max_depth": 3, "learning_rate": 0.1}]
    results = cv_search(X, y, grid, n_splits=3, max_rounds=300)
    assert len(results) == 2
    assert {"cv_auc", "cv_auc_std", "best_rounds"} <= set(results.columns)
    assert (results["best_rounds"] < 300).all(), "early stopping must stop before the cap"
    assert results["cv_auc"].is_monotonic_decreasing  # best first


def test_best_params_of_the_search_can_be_refitted(data) -> None:
    from credit_risk.train import PARAM_KEYS, best_params

    X, y = data
    grid = [
        {"max_depth": 2, "learning_rate": 0.1, "min_child_weight": 1, "reg_lambda": 1, "gamma": 0}
    ]
    best = best_params(cv_search(X, y, grid, n_splits=2, max_rounds=50))
    assert isinstance(best["max_depth"], int)
    params = {k: v for k, v in best.items() if k in PARAM_KEYS}
    fit_challenger(X, y, params, n_rounds=best["best_rounds"])  # must not raise


def test_fit_challenger_predicts_probabilities_that_rank(data) -> None:
    X, y = data
    model = fit_challenger(X, y, {"max_depth": 3, "learning_rate": 0.1}, n_rounds=100)
    assert isinstance(model, ChallengerModel)
    p = model.predict_pd(X)
    assert ((p > 0) & (p < 1)).all()
    assert gini(y, p) > 0.4


def test_scale_pos_weight_inflates_the_probabilities(data) -> None:
    """With scale_pos_weight, bads weigh about 11x more: raw PDs are far above the true rate."""
    X, y = data
    model = fit_challenger(X, y, {"max_depth": 3, "learning_rate": 0.1}, n_rounds=100)
    assert model.predict_pd(X).mean() > 2 * y.mean()


# ---------- calibration ----------


@pytest.fixture
def overconfident() -> tuple[np.ndarray, np.ndarray]:
    """Raw scores that rank well but are 3x too high, like a boosted model with class weights."""
    rng = np.random.default_rng(5)
    true_pd = rng.uniform(0.01, 0.3, size=5_000)
    y = (rng.random(5_000) < true_pd).astype(int)
    return np.clip(true_pd * 3, 0.001, 0.999), y


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_brings_the_mean_back_to_the_default_rate(overconfident, method) -> None:
    raw, y = overconfident
    calibrated = Calibrator(method).fit(raw, y).predict(raw)
    assert abs(calibrated.mean() - y.mean()) < 0.01
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_calibration_keeps_the_ranking(overconfident) -> None:
    raw, y = overconfident
    calibrated = Calibrator("sigmoid").fit(raw, y).predict(raw)
    assert gini(y, calibrated) == pytest.approx(gini(y, raw), abs=1e-9)


def test_oof_calibration_never_scores_a_row_with_a_calibrator_that_saw_it(overconfident) -> None:
    raw, y = overconfident
    oof = oof_calibrate(raw, y, "isotonic", n_splits=5)
    in_sample = Calibrator("isotonic").fit(raw, y).predict(raw)
    assert len(oof) == len(raw)
    assert not np.allclose(oof, in_sample), "out-of-fold must differ from in-sample"


def test_compare_calibrations_lowers_the_brier(overconfident) -> None:
    raw, y = overconfident
    table = compare_calibrations(raw, y)
    assert list(table.index) == ["raw", "sigmoid", "isotonic"]
    assert table.loc["raw", "brier"] > table.loc[["sigmoid", "isotonic"], "brier"].max()


def test_reliability_table(overconfident) -> None:
    raw, y = overconfident
    table = reliability_table(y, raw, n_bins=10)
    assert len(table) == 10 and table["count"].sum() == len(y)
    assert (table["mean_predicted"] > table["observed_rate"]).all()  # 3x too high everywhere
