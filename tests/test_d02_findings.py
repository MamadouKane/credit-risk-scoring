"""Tests for docs/eda_findings.md: a short list of findings, each with a number and a decision."""

from __future__ import annotations

import re

import pytest

from credit_risk.config import ROOT

FINDINGS = ROOT / "docs" / "eda_findings.md"


@pytest.fixture(scope="module")
def items() -> list[str]:
    """One string per numbered finding (from "1." to the next number)."""
    text = FINDINGS.read_text(encoding="utf-8")
    return re.findall(r"^\d+\.\s.*?(?=^\d+\.\s|\Z)", text, flags=re.MULTILINE | re.DOTALL)


def test_file_exists() -> None:
    assert FINDINGS.exists(), "write docs/eda_findings.md"


def test_between_5_and_10_findings(items: list[str]) -> None:
    assert 5 <= len(items) <= 10, f"{len(items)} findings: keep the 5 to 10 that matter"


def test_every_item_has_a_finding_and_a_decision(items: list[str]) -> None:
    for item in items:
        assert "**Finding:**" in item and "**Decision:**" in item, item[:80]


def test_every_finding_quotes_a_number(items: list[str]) -> None:
    for item in items:
        finding = item.split("**Decision:**")[0].split("**Finding:**")[-1]  # skip the item number
        assert re.search(r"\d", finding), f"no number in: {item[:80]}"


def test_no_placeholder_left() -> None:
    text = FINDINGS.read_text(encoding="utf-8")
    assert "___" not in text and "TODO" not in text
