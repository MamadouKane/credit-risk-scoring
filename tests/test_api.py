"""Tests for the scoring API, with a small scorecard trained on synthetic data (no files needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk.api.main import Applicant, Artifacts, app, get_artifacts
from credit_risk.calibrate import Calibrator
from credit_risk.scorecard import fit_scorecard


@pytest.fixture(scope="module")
def artifacts() -> Artifacts:
    rng = np.random.default_rng(11)
    n = 4_000
    X = pd.DataFrame(
        {
            "EXT_SOURCE_3": rng.uniform(0, 1, n),
            "LATE_SHARE": rng.uniform(0, 0.5, n),
            "OCCUPATION_TYPE": rng.choice(["Accountants", "Drivers", "Laborers"], n),
        }
    )
    logit = (
        -1.5
        - 3 * X["EXT_SOURCE_3"]
        + 3 * X["LATE_SHARE"]
        + 0.8 * (X["OCCUPATION_TYPE"] == "Drivers")
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).to_numpy().astype(int)
    model = fit_scorecard(X, y, max_features=3)
    assert len(model.features) == 3
    calibrator = Calibrator("sigmoid").fit(model.predict_pd(X), y)
    return Artifacts(scorecard=model, calibrator=calibrator, threshold=0.10)


@pytest.fixture
def client(artifacts) -> TestClient:
    app.dependency_overrides[get_artifacts] = lambda: artifacts
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_score_returns_the_full_decision(client) -> None:
    response = client.post(
        "/score", json={"EXT_SOURCE_3": 0.05, "LATE_SHARE": 0.4, "OCCUPATION_TYPE": "Drivers"}
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"pd", "score", "grade", "decision", "threshold", "reasons"}
    assert 0 < body["pd"] < 1 and body["grade"] in "ABCDE"
    assert body["decision"] == "reject"  # very low external score and many late payments
    assert len(body["reasons"]) == 3 and body["reasons"][0]["text"]


def test_safe_applicant_is_approved(client) -> None:
    body = client.post("/score", json={"EXT_SOURCE_3": 0.95, "LATE_SHARE": 0.0}).json()
    assert body["decision"] == "approve"


def test_missing_fields_are_allowed(client) -> None:
    """Every field is optional: a missing value falls in the missing bin, like in training."""
    assert client.post("/score", json={}).status_code == 200


def test_invalid_input_is_rejected(client) -> None:
    response = client.post("/score", json={"EXT_SOURCE_3": "not a number"})
    assert response.status_code == 422


def test_schema_covers_every_variable_of_the_trained_scorecard() -> None:
    from credit_risk.config import ROOT

    model_file = ROOT / "models" / "scorecard.joblib"
    if not model_file.exists():
        pytest.skip("no trained scorecard (run make train)")
    import joblib

    missing = set(joblib.load(model_file).features) - set(Applicant.model_fields)
    assert not missing, f"add {missing} to the Applicant schema"
