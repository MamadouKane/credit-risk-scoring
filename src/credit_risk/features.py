"""Feature engineering: affordability ratios, ages, and one row of history per applicant."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import project_path
from credit_risk.data import clean, load_table

DAYS_PER_YEAR = 365.25


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, NaN (never inf) when the denominator is 0 or missing."""
    # Divide by the denominator where its zeros are replaced by np.nan.
    # x / 0 gives inf in pandas, and one inf breaks binning, scaling and XGBoost alike.
    return numerator / denominator.replace(0, np.nan)


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add the affordability ratios a credit analyst checks first.

    CREDIT_INCOME_RATIO: loan size relative to yearly income.
    ANNUITY_INCOME_RATIO: share of income spent on this loan each year (debt-to-income proxy).
    CREDIT_GOODS_RATIO: credit above the price of the goods means fees or extra cash.
    ANNUITY_CREDIT_RATIO: yearly repayment rate, i.e. the inverse of the loan duration.
    """
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = _safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_GOODS_RATIO"] = _safe_divide(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
    df["ANNUITY_CREDIT_RATIO"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    return df


def add_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Replace DAYS_BIRTH and DAYS_EMPLOYED (negative days) by AGE_YEARS and EMPLOYED_YEARS.

    The raw columns are dropped: keeping both would add an exact duplicate (correlation -1).
    Expects clean() to have run, so the DAYS_EMPLOYED placeholder is already NaN.
    """
    df = df.copy()
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / DAYS_PER_YEAR
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / DAYS_PER_YEAR
    return df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])


def installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR: how the applicant repaid previous Home Credit loans.

    LATE_SHARE: share of instalments paid after the due date.
    MAX_DAYS_LATE: worst delay in days (0 if never late).
    UNDERPAID_SHARE: share of instalments paid below the amount due.
    N_PREV_LOANS: number of previous loans.
    An instalment not paid yet (no payment date) is neither late nor underpaid here.
    """
    days_late = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    rows = pd.DataFrame(
        {
            "SK_ID_CURR": inst["SK_ID_CURR"],
            "SK_ID_PREV": inst["SK_ID_PREV"],
            "IS_LATE": days_late > 0,
            "DAYS_LATE": days_late.fillna(0),
            "IS_UNDERPAID": inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"],
        }
    )
    # Why a new DataFrame and not new columns on `inst`: never modify the caller's 13.6M rows.
    return rows.groupby("SK_ID_CURR").agg(
        LATE_SHARE=("IS_LATE", "mean"),
        MAX_DAYS_LATE=("DAYS_LATE", "max"),
        UNDERPAID_SHARE=("IS_UNDERPAID", "mean"),
        N_PREV_LOANS=("SK_ID_PREV", "nunique"),
    )


def bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR: the applicant's credits at other institutions.

    N_BUREAU: number of credits ever reported. N_ACTIVE: credits still running.
    ANY_OVERDUE: 1 if any credit has an overdue amount. BUREAU_DEBT: debt on active credits.
    YEARS_SINCE_LAST_CREDIT: how recently the applicant took a credit (credit hunger).
    DAYS_CREDIT_UPDATE is never used: 17 records were updated after the application (leakage).
    """
    is_active = bureau["CREDIT_ACTIVE"].eq("Active")
    active_debt = bureau["AMT_CREDIT_SUM_DEBT"].where(is_active)
    years_since = -bureau["DAYS_CREDIT"] / DAYS_PER_YEAR
    rows = pd.DataFrame(
        {
            "SK_ID_CURR": bureau["SK_ID_CURR"],
            "SK_ID_BUREAU": bureau["SK_ID_BUREAU"],
            "IS_ACTIVE": is_active,
            "HAS_OVERDUE": bureau["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0,
            "ACTIVE_DEBT": active_debt,
            "YEARS_SINCE": years_since,
        }
    )
    out = rows.groupby("SK_ID_CURR").agg(
        N_BUREAU=("SK_ID_BUREAU", "size"),
        N_ACTIVE=("IS_ACTIVE", "sum"),
        ANY_OVERDUE=("HAS_OVERDUE", "max"),
        BUREAU_DEBT=("ACTIVE_DEBT", "sum"),
        YEARS_SINCE_LAST_CREDIT=("YEARS_SINCE", "min"),
    )
    out["ANY_OVERDUE"] = out["ANY_OVERDUE"].astype(int)
    return out


def build_features(app: pd.DataFrame, bureau: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Modelling table: one row per application = cleaned application + ratios + ages + history.

    History aggregates are joined with a LEFT join (applicants without history keep their row),
    they stay NaN when there is no history (their own bin in the scorecard, never 0),
    and two flags HAS_BUREAU / HAS_INSTALMENTS say whether a history exists.
    """
    out = add_ages(add_ratios(app))
    history = [
        (bureau_features(bureau), "HAS_BUREAU"),
        (installments_features(inst), "HAS_INSTALMENTS"),
    ]
    for table, flag in history:
        out[flag] = out["SK_ID_CURR"].isin(table.index).astype(int)
        out = out.merge(
            table, left_on="SK_ID_CURR", right_index=True, how="left", validate="one_to_one"
        )
    out["BUREAU_DEBT_INCOME_RATIO"] = _safe_divide(out["BUREAU_DEBT"], out["AMT_INCOME_TOTAL"])
    return out


def main() -> None:
    """Build the feature tables for application_train and application_test (make features)."""
    bureau, inst = load_table("bureau"), load_table("installments_payments")
    processed = project_path("processed")
    processed.mkdir(parents=True, exist_ok=True)
    for name in ("application_train", "application_test"):
        table = build_features(clean(load_table(name)), bureau, inst)
        table.to_parquet(processed / f"{name}_features.parquet", index=False)
        print(f"{name}: {table.shape}")


if __name__ == "__main__":
    main()
