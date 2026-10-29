"""Explain decisions: reason codes from the scorecard, SHAP for the XGBoost challenger, fairness.

Reason codes (adverse action reasons): for each variable, the points the applicant lost compared
with the best bin of that variable. The three biggest losses are the three main reasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from credit_risk.challenger import ChallengerModel
from credit_risk.scorecard import FACTOR, OFFSET, ScorecardModel

# Plain language, as a bank would write it to a rejected applicant.
REASON_TEXT = {
    "EXT_SOURCE_1": "Low score from external credit data source 1",
    "EXT_SOURCE_2": "Low score from external credit data source 2",
    "EXT_SOURCE_3": "Low score from external credit data source 3",
    "EMPLOYED_YEARS": "Short time with the current employer",
    "AGE_YEARS": "Short credit and employment history",
    "YEARS_SINCE_LAST_CREDIT": "A new credit was opened very recently",
    "OCCUPATION_TYPE": "Occupation associated with less stable income",
    "ORGANIZATION_TYPE": "Employer sector associated with higher repayment risk",
    "NAME_INCOME_TYPE": "Type of income associated with higher repayment risk",
    "NAME_EDUCATION_TYPE": "Education level associated with higher repayment risk",
    "CREDIT_GOODS_RATIO": "Loan amount well above the price of the goods financed",
    "CREDIT_INCOME_RATIO": "Loan amount high compared with income",
    "ANNUITY_INCOME_RATIO": "Repayments high compared with income",
    "ANNUITY_CREDIT_RATIO": "Repayment schedule of the loan",
    "AMT_GOODS_PRICE": "Price of the goods financed",
    "AMT_CREDIT": "Amount of the loan",
    "LATE_SHARE": "Late payments on previous loans",
    "MAX_DAYS_LATE": "Long payment delay on a previous loan",
    "UNDERPAID_SHARE": "Instalments paid below the amount due on previous loans",
    "N_ACTIVE": "Number of credits currently running at other lenders",
    "ANY_OVERDUE": "Overdue amount on a credit at another lender",
    "BUREAU_DEBT_INCOME_RATIO": "Existing debt high compared with income",
    "REGION_RATING_CLIENT_W_CITY": "Region of residence associated with higher risk",
    "DAYS_LAST_PHONE_CHANGE": "Recent change of phone number",
    "DAYS_ID_PUBLISH": "Identity document issued recently",
    "DAYS_REGISTRATION": "Recent change of registration details",
    "FLOORSMAX_AVG": "Housing information missing or associated with higher risk",
    "FLAG_DOCUMENT_3": "Supporting document 3 not provided",
}


def best_points(model: ScorecardModel) -> pd.Series:
    """Highest points each variable can give (its safest bin, missing bin included)."""
    base = (OFFSET - FACTOR * model.logistic.intercept_[0]) / len(model.features)
    best = {}
    for name, coef in zip(model.features, model.logistic.coef_[0], strict=True):
        table = model.binning.get_binned_variable(name).binning_table.build()
        bins = table.drop(index="Totals")
        woe = bins.loc[bins["Count"].astype(float) > 0, "WoE"].astype(float)
        best[name] = float((base - FACTOR * coef * woe).max())
    return pd.Series(best)


def reason_codes(model: ScorecardModel, X: pd.DataFrame, n: int = 3) -> list[list[dict]]:
    """For each applicant, the `n` variables where the most points were lost, in plain language."""
    lost = best_points(model) - model.points(X)
    out = []
    for _, row in lost.iterrows():
        top = row.sort_values(ascending=False).head(n)
        out.append(
            [
                {
                    "feature": name,
                    "text": REASON_TEXT.get(name, name.replace("_", " ").capitalize()),
                    "points_lost": round(float(value), 1),
                }
                for name, value in top.items()
            ]
        )
    return out


def shap_contributions(model: ChallengerModel, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP values computed by XGBoost itself, on the log-odds scale.

    Each column says how much a feature pushed the applicant's log-odds of default up (>0) or
    down (<0) from the average (BIAS). The row sum is the model output.
    """
    matrix = xgb.DMatrix(model.prepared(X), enable_categorical=True)
    values = model.model.get_booster().predict(matrix, pred_contribs=True)
    return pd.DataFrame(values, columns=[*X.columns, "BIAS"], index=X.index)


def fairness_table(group: pd.Series, approved: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per group: size, approval rate, default rate among the approved, and the approval ratio
    to the most approved group (the "four-fifths rule": a ratio below 0.8 deserves a review)."""
    frame = pd.DataFrame(
        {"group": np.asarray(group), "approved": np.asarray(approved), "y": np.asarray(y)}
    )
    table = frame.groupby("group").agg(count=("y", "size"), approval_rate=("approved", "mean"))
    table["default_rate_approved"] = frame[frame["approved"]].groupby("group")["y"].mean()
    table["approval_ratio"] = table["approval_rate"] / table["approval_rate"].max()
    return table
