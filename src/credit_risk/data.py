"""Load the Home Credit tables. Each CSV is parsed once, then served from a parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk.config import project_path


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
    desc = pd.read_csv(path, encoding='cp1252', index_col=0)
    # Return a clean 0-based index (the file's row numbers start at 1).
    return desc.reset_index(drop=True)
