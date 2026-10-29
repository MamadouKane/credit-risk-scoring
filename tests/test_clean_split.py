"""Tests for clean() (data-quality fixes from the EDA) and split() (leak-free 70/15/15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import clean, load_table, split

PLACEHOLDER = 365243


@pytest.fixture
def raw() -> pd.DataFrame:
    """Five applications with every issue clean() must handle."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5],
            "TARGET": [0, 1, 0, 0, 1],
            "DAYS_EMPLOYED": [-1000, PLACEHOLDER, -200, PLACEHOLDER, -50],
            "CODE_GENDER": ["F", "M", "XNA", "F", "M"],
            "ORGANIZATION_TYPE": ["Bank", "XNA", "Trade", "XNA", "School"],
            "NAME_FAMILY_STATUS": ["Married", "Unknown", "Single", "Widow", "Married"],
            "APARTMENTS_AVG": [0.1, np.nan, 0.3, 0.2, np.nan],
            "APARTMENTS_MODE": [0.1, np.nan, 0.3, 0.2, np.nan],
            "APARTMENTS_MEDI": [0.1, np.nan, 0.3, 0.2, np.nan],
            "FONDKAPREMONT_MODE": ["a", None, "b", "a", None],  # no _AVG twin: must stay
        }
    )


# ---------- clean ----------


def test_placeholder_becomes_nan(raw: pd.DataFrame) -> None:
    out = clean(raw)
    assert not (out["DAYS_EMPLOYED"] == PLACEHOLDER).any()
    assert out["DAYS_EMPLOYED"].isna().sum() == 2


def test_placeholder_flag_marks_exactly_those_rows(raw: pd.DataFrame) -> None:
    out = clean(raw)
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1, 0, 1, 0]


def test_missing_codes_become_nan(raw: pd.DataFrame) -> None:
    out = clean(raw)
    codes = {"CODE_GENDER": "XNA", "ORGANIZATION_TYPE": "XNA", "NAME_FAMILY_STATUS": "Unknown"}
    for col, code in codes.items():
        assert not (out[col] == code).any(), col
    assert out["CODE_GENDER"].isna().sum() == 1


def test_building_twins_are_dropped_but_avg_and_lone_mode_kept(raw: pd.DataFrame) -> None:
    out = clean(raw)
    assert "APARTMENTS_AVG" in out
    assert "APARTMENTS_MODE" not in out and "APARTMENTS_MEDI" not in out
    assert "FONDKAPREMONT_MODE" in out


def test_clean_does_not_modify_its_input(raw: pd.DataFrame) -> None:
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clean_works_without_target(raw: pd.DataFrame) -> None:
    out = clean(raw.drop(columns="TARGET"))  # application_test has no TARGET
    assert len(out) == 5


# ---------- split ----------


@pytest.fixture
def population() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 10_000
    return pd.DataFrame({"SK_ID_CURR": np.arange(n), "TARGET": (rng.random(n) < 0.08).astype(int)})


def test_split_sizes_are_70_15_15(population: pd.DataFrame) -> None:
    train, calib, test = split(population)
    assert (len(train), len(calib), len(test)) == (7_000, 1_500, 1_500)


def test_split_parts_are_disjoint_and_complete(population: pd.DataFrame) -> None:
    train, calib, test = split(population)
    ids = [set(part["SK_ID_CURR"]) for part in (train, calib, test)]
    overlap = ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]
    assert not overlap, "an application is in two parts"
    assert set.union(*ids) == set(population["SK_ID_CURR"])


def test_split_is_stratified(population: pd.DataFrame) -> None:
    rate = population["TARGET"].mean()
    for part in split(population):
        assert abs(part["TARGET"].mean() - rate) < 0.002


def test_split_is_reproducible(population: pd.DataFrame) -> None:
    first, second = split(population, seed=42), split(population, seed=42)
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)
    other = split(population, seed=7)
    assert not first[0]["SK_ID_CURR"].equals(other[0]["SK_ID_CURR"])


# ---------- real data ----------


@pytest.mark.data
def test_real_clean() -> None:
    out = clean(load_table("application_train"))
    assert out.shape == (307_511, 95)
    assert out["DAYS_EMPLOYED_ANOM"].sum() == 55_374
    assert out["ORGANIZATION_TYPE"].isna().sum() == 55_374


@pytest.mark.data
def test_real_split() -> None:
    train, calib, test = split(clean(load_table("application_train")))
    assert (len(train), len(calib), len(test)) == (215_257, 46_127, 46_127)
    for part in (train, calib, test):
        assert round(part["TARGET"].mean(), 4) == 0.0807
