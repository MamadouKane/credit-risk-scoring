"""Load the Home Credit tables. Each CSV is parsed once, then served from a parquet cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import load_config, project_path

MISSING_CODES = ["XNA", "Unknown"]  # how Home Credit writes "missing" in text columns


def load_table(
    name: str,
    raw_dir: Path | None = None,
    interim_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return the table `name` (e.g. "bureau") from data/raw/<name>.csv.

    The first call parses the CSV and writes data/interim/<name>.parquet.
    Later calls read the parquet file: it is typed, compressed and about 10x faster to load.
    """
    raw_dir = raw_dir or project_path("raw")
    interim_dir = interim_dir or project_path("interim")
    # Path of the cache file: <interim_dir>/<name>.parquet
    cache = interim_dir / f"{name}.parquet"

    # Return the cached table when caching is on and the file exists.
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    csv = raw_dir / f"{name}.csv"
    # Fail early with a clear message when the CSV does not exist.
    # A wrong table name must not end in a cryptic pandas error.
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found: run `make data` or check the name '{name}'")
    df = pd.read_csv(csv)

    if use_cache:
        # Create interim_dir if needed, then write df to the cache (no index).
        interim_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def load_column_descriptions(path: Path | None = None) -> pd.DataFrame:
    """Return the Kaggle column dictionary (Table, Row, Description, Special).

    The file is encoded in cp1252 (Windows), not UTF-8: byte 0x85 is the "..." character.
    """
    path = path or project_path("column_descriptions")
    # Read the CSV with the right encoding; its first column is a row number.
    desc = pd.read_csv(path, encoding="cp1252", index_col=0)
    # Return a clean 0-based index (the file's row numbers start at 1).
    return desc.reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data-quality issues found in the EDA. Returns a new DataFrame.

    1. DAYS_EMPLOYED = 365243 codes "no employer" (pensioners): NaN + flag DAYS_EMPLOYED_ANOM.
    2. Text codes for "missing" (XNA, Unknown) become real missing values.
    3. Building features exist 3 times (_AVG, _MODE, _MEDI): keep _AVG only.
    Works on application_test too (no TARGET needed).
    """
    # Work on a copy: a cleaning function must never modify the caller's data.
    df = df.copy()

    placeholder = load_config()["data"]["days_employed_placeholder"]
    # Boolean mask of the placeholder rows.
    is_placeholder = df["DAYS_EMPLOYED"] == placeholder
    # Flag as 0/1 integers, then replace the placeholder by NaN.
    # Why the flag: placeholder rows default less (5.40 % vs 8.66 %), NaN alone would lose that.
    df["DAYS_EMPLOYED_ANOM"] = is_placeholder.astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].mask(is_placeholder, np.nan)

    text_cols = df.select_dtypes(exclude="number").columns
    # Replace every code of MISSING_CODES by np.nan in the text columns.
    df[text_cols] = df[text_cols].replace(MISSING_CODES, np.nan)

    # Building twins: names ending with _MODE or _MEDI whose _AVG version exists.
    # FONDKAPREMONT_MODE has no _AVG twin and must stay.
    twins = [
        col
        for col in df.columns
        if (col.endswith(("_MODE", "_MEDI"))) and (col.rsplit("_", 1)[0] + "_AVG" in df.columns)
    ]
    return df.drop(columns=twins)


def split(
    df: pd.DataFrame, target: str = "TARGET", seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train / calibration / test (70 / 15 / 15).

    Train fits the models, calibration fits the calibrator and the decision threshold,
    test is touched once at the end. Stratifying keeps the default rate equal in all three.
    """
    sizes = load_config()["split"]
    holdout = sizes["calibration"] + sizes["test"]  # 0.30
    # First cut: train (70 %) vs the rest (30 %), stratified on the target.
    train, rest = train_test_split(df, test_size=holdout, stratify=df[target], random_state=seed)
    # Second cut: `rest` into calibration and test (15 % + 15 % of the total).
    calib, test = train_test_split(
        rest, test_size=sizes["test"] / holdout, stratify=rest[target], random_state=seed
    )
    return train, calib, test
