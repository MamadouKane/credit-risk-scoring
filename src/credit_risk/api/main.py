"""Scoring API: POST /score returns PD, points, grade, decision and the top 3 reasons.

Run locally with `make serve`, then open http://127.0.0.1:8000/docs (Swagger UI).
The champion (calibrated scorecard) decides: its reasons can be shown to a rejected applicant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, ConfigDict, Field

from credit_risk.calibrate import Calibrator
from credit_risk.config import project_path
from credit_risk.explain import reason_codes
from credit_risk.scorecard import ScorecardModel


class Applicant(BaseModel):
    """The 20 scorecard variables, as built by features.py. Every field is optional:
    a missing value is scored with the WoE of the missing bin, exactly as in training."""

    model_config = ConfigDict(extra="forbid")

    EXT_SOURCE_1: float | None = Field(None, ge=0, le=1, description="External credit score 1")
    EXT_SOURCE_2: float | None = Field(None, ge=0, le=1, description="External credit score 2")
    EXT_SOURCE_3: float | None = Field(
        None, ge=0, le=1, description="External credit score 3", examples=[0.35]
    )
    EMPLOYED_YEARS: float | None = Field(None, ge=0, description="Years with the current employer")
    YEARS_SINCE_LAST_CREDIT: float | None = Field(
        None, ge=0, description="Years since the last external credit"
    )
    OCCUPATION_TYPE: str | None = Field(None, examples=["Laborers"])
    ORGANIZATION_TYPE: str | None = Field(None, examples=["Business Entity Type 3"])
    NAME_EDUCATION_TYPE: str | None = Field(None, examples=["Secondary / secondary special"])
    CREDIT_GOODS_RATIO: float | None = Field(
        None, ge=0, description="Credit amount / price of the goods"
    )
    AMT_GOODS_PRICE: float | None = Field(None, ge=0)
    LATE_SHARE: float | None = Field(
        None, ge=0, le=1, description="Share of late instalments", examples=[0.1]
    )
    MAX_DAYS_LATE: float | None = Field(None, ge=0)
    N_ACTIVE: float | None = Field(None, ge=0, description="Active credits at other lenders")
    BUREAU_DEBT_INCOME_RATIO: float | None = Field(None, ge=0)
    REGION_RATING_CLIENT_W_CITY: float | None = Field(None, ge=1, le=3)
    DAYS_LAST_PHONE_CHANGE: float | None = Field(
        None, le=0, description="Days, negative = in the past"
    )
    DAYS_ID_PUBLISH: float | None = Field(None, le=0)
    DAYS_REGISTRATION: float | None = Field(None, le=0)
    FLOORSMAX_AVG: float | None = Field(None, ge=0, le=1)
    FLAG_DOCUMENT_3: int | None = Field(None, ge=0, le=1)


class Reason(BaseModel):
    feature: str
    text: str
    points_lost: float


class ScoreResponse(BaseModel):
    pd: float = Field(description="Calibrated probability of early payment difficulty")
    score: float = Field(description="Scorecard points (600 = odds 50:1, +20 points = odds x2)")
    grade: str
    decision: str = Field(description='"approve" if pd < threshold, else "reject"')
    threshold: float
    reasons: list[Reason] = Field(description="The 3 variables that cost the most points")


@dataclass
class Artifacts:
    scorecard: ScorecardModel
    calibrator: Calibrator
    threshold: float


@lru_cache
def get_artifacts() -> Artifacts:
    """Load the trained models once (first request)."""
    models = project_path("models")
    decision = json.loads((models / "decision.json").read_text())
    return Artifacts(
        scorecard=joblib.load(models / "scorecard.joblib"),
        calibrator=joblib.load(models / "calibrators.joblib")["scorecard"],
        threshold=decision["scorecard"]["threshold"],
    )


def to_frame(applicant: Applicant, model: ScorecardModel) -> pd.DataFrame:
    """One row with every variable the binning was fitted on (absent ones are missing)."""
    values = applicant.model_dump()
    row = {name: values.get(name, np.nan) for name in model.binning.variable_names}
    frame = pd.DataFrame([row])
    for name in model.binning.categorical_variables or []:
        frame[name] = frame[name].astype("object")
    numeric = [c for c in frame.columns if c not in (model.binning.categorical_variables or [])]
    frame[numeric] = frame[numeric].astype(float)
    return frame


app = FastAPI(
    title="Credit risk scoring",
    description="Application scoring: PD at origination, points, grade, cost-based decision.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(
    applicant: Applicant, artifacts: Annotated[Artifacts, Depends(get_artifacts)]
) -> ScoreResponse:
    model = artifacts.scorecard
    X = to_frame(applicant, model)
    pd_hat = float(artifacts.calibrator.predict(model.predict_pd(X))[0])
    return ScoreResponse(
        pd=round(pd_hat, 4),
        score=round(float(model.score(X)[0]), 1),
        grade=str(model.grade(X)[0]),
        decision="approve" if pd_hat < artifacts.threshold else "reject",
        threshold=round(artifacts.threshold, 4),
        reasons=[Reason(**r) for r in reason_codes(model, X)[0]],
    )
