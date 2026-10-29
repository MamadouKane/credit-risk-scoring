"""Drift monitoring with the Population Stability Index (PSI).

PSI = sum over bins of (a - e) * ln(a / e), with e the share of the reference population (train)
and a the share of the new population in each bin. Bins are the deciles of the reference;
missing values get their own bin. Usual reading in banking:
below 0.1 stable, 0.1 to 0.25 to watch, above 0.25 significant shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4  # an empty bin on one side would give log(0)


def _psi_from_shares(e: pd.Series, a: pd.Series) -> float:
    idx = e.index.union(a.index)
    e = e.reindex(idx, fill_value=0) + EPS
    a = a.reindex(idx, fill_value=0) + EPS
    return float(((a - e) * np.log(a / e)).sum())


def psi(expected: pd.Series, actual: pd.Series, q: int = 10) -> float:
    """PSI of a numeric variable, decile bins of the reference, open-ended first and last bins."""
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, q + 1)))
    edges[0], edges[-1] = -np.inf, np.inf  # values outside the training range still land in a bin

    def shares(x: pd.Series) -> pd.Series:
        x = pd.Series(np.asarray(x, dtype=float))
        bins = pd.cut(x, edges).astype(str).where(x.notna(), "missing")
        return bins.value_counts(normalize=True)

    return _psi_from_shares(shares(expected), shares(actual))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI of a categorical variable: each category is a bin (missing included)."""

    def shares(x: pd.Series) -> pd.Series:
        return pd.Series(x).astype("object").fillna("missing").value_counts(normalize=True)

    return _psi_from_shares(shares(expected), shares(actual))


def status(value: float) -> str:
    if value < 0.1:
        return "stable"
    return "to watch" if value < 0.25 else "significant shift"


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, variables: list[str]
) -> pd.DataFrame:
    """One row per variable: PSI and its reading, biggest drift first."""
    rows = []
    for name in variables:
        numeric = pd.api.types.is_numeric_dtype(reference[name])
        value = (psi if numeric else psi_categorical)(reference[name], current[name])
        rows.append({"variable": name, "psi": value, "status": status(value)})
    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    return report.reset_index(drop=True)


def shifted_batch(
    reference: pd.DataFrame, multipliers: dict[str, float], n: int = 10_000, seed: int = 0
) -> pd.DataFrame:
    """Simulated "production" batch: a sample of the reference with some columns multiplied
    (e.g. incomes +30 %), to show that the report catches a known shift."""
    batch = reference.sample(n=min(n, len(reference)), random_state=seed).copy()
    for name, factor in multipliers.items():
        batch[name] = batch[name] * factor
    return batch
