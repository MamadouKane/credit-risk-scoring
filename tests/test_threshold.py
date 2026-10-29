"""Tests for the cost-based decision threshold and the expected-loss view per grade."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.threshold import (
    breakeven_threshold,
    cost_curve,
    decision_summary,
    expected_cost,
    expected_loss_by_grade,
    optimal_threshold,
    sensitivity,
)


@pytest.fixture
def book() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4 applications of 100,000: two goods, two bads, PDs from safe to risky."""
    y = np.array([0, 0, 1, 1])
    pd_hat = np.array([0.02, 0.08, 0.15, 0.40])
    ead = np.full(4, 100_000.0)
    return y, pd_hat, ead


def test_expected_cost_counts_both_errors(book) -> None:
    y, pd_hat, ead = book
    # threshold 0.10: approve the first two (goods), reject the bads -> no error, cost 0
    assert expected_cost(y, pd_hat, ead, 0.10, lgd=0.45, margin=0.05) == 0
    # threshold 0.20: the bad at 0.15 is approved -> loses LGD x EAD = 45,000
    assert expected_cost(y, pd_hat, ead, 0.20, lgd=0.45, margin=0.05) == pytest.approx(45_000)
    # threshold 0.05: the good at 0.08 is rejected -> lost margin 5,000
    assert expected_cost(y, pd_hat, ead, 0.05, lgd=0.45, margin=0.05) == pytest.approx(5_000)


def test_cost_curve_and_optimal_threshold(book) -> None:
    y, pd_hat, ead = book
    curve = cost_curve(y, pd_hat, ead, lgd=0.45, margin=0.05, thresholds=np.linspace(0.01, 0.5, 50))
    assert {"threshold", "cost", "approval_rate"} <= set(curve.columns)
    best = optimal_threshold(curve)
    assert 0.08 < best <= 0.15 and curve["cost"].min() == 0


def test_breakeven_threshold_is_margin_over_lgd_plus_margin() -> None:
    """Approve when PD x LGD < (1 - PD) x margin  <=>  PD < margin / (LGD + margin)."""
    assert breakeven_threshold(lgd=0.45, margin=0.05) == pytest.approx(0.10)
    assert breakeven_threshold(lgd=0.60, margin=0.05) < breakeven_threshold(lgd=0.30, margin=0.05)


def test_sensitivity_threshold_falls_when_lgd_rises() -> None:
    rng = np.random.default_rng(0)
    n = 200_000  # the cost curve is flat near its minimum: a large sample keeps noise low
    pd_hat = rng.uniform(0.01, 0.5, n)
    y = (rng.random(n) < pd_hat).astype(int)  # calibrated by construction
    ead = rng.uniform(50_000, 500_000, n)
    table = sensitivity(y, pd_hat, ead, lgd_grid=[0.3, 0.45, 0.6], margin_grid=[0.05])
    assert table["optimal_threshold"].is_monotonic_decreasing
    # on calibrated PDs, the empirical optimum sits close to the break-even formula
    np.testing.assert_allclose(table["optimal_threshold"], table["breakeven"], atol=0.015)


def test_decision_summary_reports_money_and_rates(book) -> None:
    y, pd_hat, ead = book
    out = decision_summary(y, pd_hat, ead, threshold=0.10, lgd=0.45, margin=0.05)
    assert out["approval_rate"] == 0.5
    assert out["default_rate_approved"] == 0.0
    assert out["cost_approve_all"] == pytest.approx(90_000)  # both bads approved
    assert out["saving_per_1000"] == pytest.approx(90_000 / 4 * 1000)


def test_expected_loss_by_grade() -> None:
    grades = pd.Series(["A", "A", "B"])
    out = expected_loss_by_grade(
        grades,
        np.array([0.01, 0.03, 0.10]),
        np.array([100.0, 300.0, 200.0]),
        np.array([0, 0, 1]),
        lgd=0.45,
    )
    assert out.loc["A", "expected_loss"] == pytest.approx(0.45 * (0.01 * 100 + 0.03 * 300))
    assert out.loc["B", "observed_default_rate"] == 1.0
    assert {"count", "mean_pd", "exposure", "el_rate"} <= set(out.columns)
