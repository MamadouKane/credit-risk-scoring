"""Tests for features.py: correct values, one row per application, no history leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import clean, load_table
from credit_risk.features import (
    add_ages,
    add_ratios,
    build_features,
    bureau_features,
    installments_features,
)


@pytest.fixture
def app() -> pd.DataFrame:
    """Three applications after clean(): client 3 has no history at all."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "AMT_INCOME_TOTAL": [100_000.0, 50_000.0, 0.0],
            "AMT_CREDIT": [300_000.0, 200_000.0, 100_000.0],
            "AMT_ANNUITY": [15_000.0, np.nan, 5_000.0],
            "AMT_GOODS_PRICE": [250_000.0, 200_000.0, np.nan],
            "DAYS_BIRTH": [-10_957, -14_610, -21_915],
            "DAYS_EMPLOYED": [-1_826.0, np.nan, -365.0],
        }
    )


@pytest.fixture
def bureau() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 12, 20],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Closed"],
            "DAYS_CREDIT": [-100, -2_000, -400, -730],
            "AMT_CREDIT_SUM_DEBT": [10_000.0, 5_000.0, np.nan, 0.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 500.0, 0.0],
            "DAYS_CREDIT_UPDATE": [-10, -1_500, 30, -700],  # 30 = updated after the application
        }
    )


@pytest.fixture
def inst() -> pd.DataFrame:
    """Client 1: 4 instalments, 1 late (10 days), 1 underpaid. Client 2: 1 on time, 1 unpaid."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 1, 2, 2],
            "SK_ID_PREV": [100, 100, 101, 101, 200, 200],
            "DAYS_INSTALMENT": [-300, -270, -60, -30, -90, -60],
            "DAYS_ENTRY_PAYMENT": [-305, -260, -60, -35, -95, np.nan],
            "AMT_INSTALMENT": [1_000.0, 1_000.0, 500.0, 500.0, 800.0, 800.0],
            "AMT_PAYMENT": [1_000.0, 1_000.0, 400.0, 500.0, 800.0, np.nan],
        }
    )


# ---------- application features ----------


def test_ratios_values(app: pd.DataFrame) -> None:
    out = add_ratios(app)
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(3.0)
    assert out.loc[0, "ANNUITY_INCOME_RATIO"] == pytest.approx(0.15)
    assert out.loc[0, "CREDIT_GOODS_RATIO"] == pytest.approx(1.2)
    assert out.loc[0, "ANNUITY_CREDIT_RATIO"] == pytest.approx(0.05)


def test_ratios_are_nan_not_inf_when_denominator_is_missing_or_zero(app: pd.DataFrame) -> None:
    out = add_ratios(app)
    assert np.isnan(out.loc[2, "CREDIT_INCOME_RATIO"])  # income 0
    assert np.isnan(out.loc[2, "CREDIT_GOODS_RATIO"])  # goods price missing
    assert np.isnan(out.loc[1, "ANNUITY_INCOME_RATIO"])  # annuity missing
    ratios = out.filter(like="_RATIO")
    assert not np.isinf(ratios.to_numpy()).any()


def test_ages_in_years_and_raw_days_dropped(app: pd.DataFrame) -> None:
    out = add_ages(app)
    assert out["AGE_YEARS"].round(1).tolist() == [30.0, 40.0, 60.0]
    assert out.loc[0, "EMPLOYED_YEARS"] == pytest.approx(5.0, abs=0.01)
    assert np.isnan(out.loc[1, "EMPLOYED_YEARS"])  # placeholder already NaN after clean()
    assert "DAYS_BIRTH" not in out and "DAYS_EMPLOYED" not in out


def test_feature_functions_do_not_modify_input(app: pd.DataFrame) -> None:
    before = app.copy()
    add_ratios(app)
    add_ages(app)
    pd.testing.assert_frame_equal(app, before)


# ---------- history aggregates ----------


def test_installments_features(inst: pd.DataFrame) -> None:
    out = installments_features(inst)
    assert out.index.name == "SK_ID_CURR" and len(out) == 2
    assert out.loc[1, "LATE_SHARE"] == pytest.approx(0.25)
    assert out.loc[1, "MAX_DAYS_LATE"] == 10
    assert out.loc[1, "UNDERPAID_SHARE"] == pytest.approx(0.25)
    assert out.loc[1, "N_PREV_LOANS"] == 2
    assert out.loc[2, "MAX_DAYS_LATE"] == 0  # the unpaid instalment is not counted as late


def test_installments_only_unpaid_gives_zero_not_nan() -> None:
    """A client whose instalments are all unpaid was never late: 0, as the docstring promises."""
    unpaid = pd.DataFrame(
        {
            "SK_ID_CURR": [7, 7],
            "SK_ID_PREV": [1, 1],
            "DAYS_INSTALMENT": [-60, -30],
            "DAYS_ENTRY_PAYMENT": [np.nan, np.nan],
            "AMT_INSTALMENT": [100.0, 100.0],
            "AMT_PAYMENT": [np.nan, np.nan],
        }
    )
    out = installments_features(unpaid)
    assert out.loc[7, "MAX_DAYS_LATE"] == 0
    assert out.loc[7, "LATE_SHARE"] == 0


def test_bureau_features(bureau: pd.DataFrame) -> None:
    out = bureau_features(bureau)
    assert out.index.name == "SK_ID_CURR" and len(out) == 2
    assert out.loc[1, "N_BUREAU"] == 3
    assert out.loc[1, "N_ACTIVE"] == 2
    assert out.loc[1, "ANY_OVERDUE"] == 1
    assert out.loc[1, "BUREAU_DEBT"] == pytest.approx(10_000.0)  # active credits only
    assert out.loc[1, "YEARS_SINCE_LAST_CREDIT"] == pytest.approx(100 / 365.25)
    assert out.loc[2, "N_ACTIVE"] == 0


def test_bureau_features_never_use_the_update_date(bureau: pd.DataFrame) -> None:
    out = bureau_features(bureau)
    assert not any("UPDATE" in col for col in out.columns)


# ---------- full table ----------


def test_build_keeps_one_row_per_application(app, bureau, inst) -> None:
    out = build_features(app, bureau, inst)
    assert len(out) == 3 and out["SK_ID_CURR"].is_unique


def test_build_flags_and_missing_history(app, bureau, inst) -> None:
    out = build_features(app, bureau, inst).set_index("SK_ID_CURR")
    assert out["HAS_BUREAU"].tolist() == [1, 1, 0]
    assert out["HAS_INSTALMENTS"].tolist() == [1, 1, 0]
    assert np.isnan(out.loc[3, "LATE_SHARE"]) and np.isnan(out.loc[3, "N_ACTIVE"])  # NaN, not 0
    assert out.loc[1, "BUREAU_DEBT_INCOME_RATIO"] == pytest.approx(0.1)


def test_build_works_without_target(app, bureau, inst) -> None:
    out = build_features(app.drop(columns="TARGET"), bureau, inst)
    assert len(out) == 3


# ---------- real data ----------


@pytest.mark.data
def test_real_features() -> None:
    app = clean(load_table("application_train"))
    out = build_features(app, load_table("bureau"), load_table("installments_payments"))
    assert out.shape == (307_511, 111)
    assert round(out["HAS_BUREAU"].mean(), 4) == 0.8569
    assert round(out["CREDIT_INCOME_RATIO"].median(), 3) == 3.265
    assert round(out["LATE_SHARE"].median(), 3) == 0.019
