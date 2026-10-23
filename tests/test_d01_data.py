"""Tests for loading the raw tables with a parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.data import load_column_descriptions, load_table


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """A raw folder with one tiny CSV, and an empty interim folder."""
    raw, interim = tmp_path / "raw", tmp_path / "interim"
    raw.mkdir()
    pd.DataFrame(
        {"SK_ID_CURR": [100001, 100002, 100003], "TARGET": [0, 1, 0], "NAME": ["a", "b", None]}
    ).to_csv(raw / "tiny.csv", index=False)
    return raw, interim


def test_load_table_reads_the_csv(dirs: tuple[Path, Path]) -> None:
    raw, interim = dirs
    df = load_table("tiny", raw_dir=raw, interim_dir=interim)
    assert df.shape == (3, 3)
    assert df["TARGET"].sum() == 1


def test_load_table_writes_a_parquet_cache(dirs: tuple[Path, Path]) -> None:
    raw, interim = dirs
    load_table("tiny", raw_dir=raw, interim_dir=interim)
    assert (interim / "tiny.parquet").exists()


def test_load_table_reads_the_cache_on_the_second_call(dirs: tuple[Path, Path]) -> None:
    raw, interim = dirs
    first = load_table("tiny", raw_dir=raw, interim_dir=interim)
    (raw / "tiny.csv").unlink()  # only the cache is left
    second = load_table("tiny", raw_dir=raw, interim_dir=interim)
    pd.testing.assert_frame_equal(first, second)


def test_load_table_without_cache_writes_nothing(dirs: tuple[Path, Path]) -> None:
    raw, interim = dirs
    load_table("tiny", raw_dir=raw, interim_dir=interim, use_cache=False)
    assert not (interim / "tiny.parquet").exists()


def test_load_table_unknown_name_raises(dirs: tuple[Path, Path]) -> None:
    raw, interim = dirs
    with pytest.raises(FileNotFoundError, match="nope"):
        load_table("nope", raw_dir=raw, interim_dir=interim)


def test_column_descriptions_are_read_as_cp1252(tmp_path: Path) -> None:
    path = tmp_path / "desc.csv"
    text = ',Table,Row,Description,Special\n1,application,AMT_CREDIT,Credit amount…,\n'
    path.write_bytes(text.encode("cp1252"))  # … becomes byte 0x85, invalid in UTF-8
    desc = load_column_descriptions(path)
    assert list(desc.columns) == ["Table", "Row", "Description", "Special"]
    assert desc.loc[0, "Description"] == "Credit amount…"


@pytest.mark.data
def test_real_application_train() -> None:
    df = load_table("application_train")
    assert df.shape == (307_511, 122)
    assert df["SK_ID_CURR"].is_unique
    assert 0.080 < df["TARGET"].mean() < 0.081


@pytest.mark.data
def test_real_column_descriptions() -> None:
    desc = load_column_descriptions()
    assert len(desc) == 219
    assert "DAYS_EMPLOYED" in set(desc["Row"])
