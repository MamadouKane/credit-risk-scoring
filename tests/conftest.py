"""Shared fixtures. Tests marked `data` are skipped when the Kaggle files are missing (CI)."""

from __future__ import annotations

import pytest

from credit_risk.config import project_path

RAW_TRAIN = project_path("raw") / "application_train.csv"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if RAW_TRAIN.exists():
        return
    skip = pytest.mark.skip(reason="real data not found in data/raw (run `make data`)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
