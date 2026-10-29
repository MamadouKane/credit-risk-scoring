"""Tests for the Population Stability Index drift report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.drift import drift_report, psi, psi_categorical, shifted_batch


@pytest.fixture
def reference() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 20_000
    return pd.DataFrame(
        {
            "INCOME": rng.lognormal(12, 0.5, n),
            "SCORE": rng.uniform(0, 1, n),
            "PRODUCT": rng.choice(["cash", "revolving"], n, p=[0.9, 0.1]),
        }
    )


def test_psi_is_zero_on_the_same_population(reference) -> None:
    assert psi(reference["SCORE"], reference["SCORE"]) == pytest.approx(0, abs=1e-9)


def test_psi_detects_a_reshaped_distribution(reference) -> None:
    other = reference["SCORE"] ** 0.7
    assert psi(reference["SCORE"], other) > 0.1


def test_psi_catches_values_outside_the_training_range(reference) -> None:
    shifted = reference["SCORE"] + 2  # every value above the training max
    assert psi(reference["SCORE"], shifted) > 0.25


def test_psi_categorical(reference) -> None:
    same = psi_categorical(reference["PRODUCT"], reference["PRODUCT"])
    only_cash = psi_categorical(reference["PRODUCT"], pd.Series(["cash"] * 1_000))
    assert same == pytest.approx(0, abs=1e-9) and only_cash > 0.1


def test_drift_report_flags_the_shifted_variables(reference) -> None:
    batch = shifted_batch(reference, {"INCOME": 1.5}, seed=1)
    report = drift_report(reference, batch, ["INCOME", "SCORE", "PRODUCT"])
    status = report.set_index("variable")["status"]
    assert status["INCOME"] == "significant shift"
    assert status["SCORE"] == "stable"
    assert list(report.columns) == ["variable", "psi", "status"]
    assert report["psi"].is_monotonic_decreasing
