"""Train both models on the training set, calibrate them and choose the threshold on the
calibration set, log every step to MLflow.

The test set is never read here: it is touched once, at the end, by `make evaluate`.
"""

from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from credit_risk.calibrate import Calibrator, compare_calibrations, oof_calibrate, reliability_table
from credit_risk.challenger import ChallengerModel, cv_search, fit_challenger
from credit_risk.config import ROOT, load_config, project_path
from credit_risk.data import split
from credit_risk.metrics import summary
from credit_risk.reports import cost_plot, reliability_plot
from credit_risk.scorecard import ScorecardModel, fit_scorecard, grade_table, iv_table
from credit_risk.threshold import breakeven_threshold, cost_curve, optimal_threshold, sensitivity

CONFIG = load_config()
MODELS = project_path("models")
FIGURES = project_path("figures")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Feature table built by `make features`, split 70 / 15 / 15."""
    features = pd.read_parquet(project_path("processed") / "application_train_features.parquet")
    return split(features, seed=CONFIG["split"]["seed"])


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model inputs (without ids, target and excluded variables) and target."""
    return df.drop(columns=CONFIG["model"]["exclude"]), df["TARGET"]


def setup_mlflow() -> None:
    db = ROOT / CONFIG["mlflow"]["tracking_uri"].removeprefix("sqlite:///")
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    mlflow.set_experiment(CONFIG["mlflow"]["experiment"])


def train_scorecard(train: pd.DataFrame, calib: pd.DataFrame) -> ScorecardModel:
    """Champion: fit the scorecard on train, report on calibration, log everything to MLflow."""
    X_train, y_train = xy(train)
    X_calib, y_calib = xy(calib)
    model = fit_scorecard(X_train, y_train.to_numpy())
    metrics = summary(y_calib, model.predict_pd(X_calib))
    grades = grade_table(model.grade(X_calib), y_calib)

    with mlflow.start_run(run_name="scorecard"):
        params = ("min_iv", "max_corr", "max_features")
        mlflow.log_params({k: CONFIG["scorecard"][k] for k in params})
        mlflow.log_param("n_features", len(model.features))
        mlflow.log_metrics({f"calib_{k}": v for k, v in metrics.items()})
        mlflow.log_text(iv_table(model.binning).to_csv(index=False), "iv_table.csv")
        mlflow.log_text(grades.to_csv(), "grades_calibration.csv")

    print("scorecard:", {k: round(v, 4) for k, v in metrics.items()})
    return model


PARAM_KEYS = ("max_depth", "learning_rate", "min_child_weight", "reg_lambda", "gamma")


def best_params(search: pd.DataFrame) -> dict:
    """Best row of the CV search as plain Python values.

    search.iloc[0] would turn every value into a float (a row has a single dtype) and XGBoost
    rejects max_depth = 6.0; to_dict("records") keeps the type of each column.
    """
    return search.to_dict("records")[0]


def train_challenger(train: pd.DataFrame, calib: pd.DataFrame) -> ChallengerModel:
    """Challenger: 5-fold CV search on train (early stopping in each fold), final refit on train."""
    X_train, y_train = xy(train)
    X_calib, y_calib = xy(calib)
    search = cv_search(X_train, y_train.to_numpy(), n_splits=CONFIG["cv"]["n_splits"])
    best = best_params(search)
    params = {k: v for k, v in best.items() if k in PARAM_KEYS}
    model = fit_challenger(X_train, y_train.to_numpy(), params, n_rounds=int(best["best_rounds"]))
    metrics = summary(y_calib, model.predict_pd(X_calib))  # raw probabilities, not calibrated yet

    with mlflow.start_run(run_name="xgboost_challenger"):
        mlflow.log_params({**params, "n_rounds": int(best["best_rounds"])})
        mlflow.log_metrics({"cv_auc": best["cv_auc"], "cv_auc_std": best["cv_auc_std"]})
        mlflow.log_metrics({f"calib_raw_{k}": v for k, v in metrics.items()})
        mlflow.log_text(search.to_csv(index=False), "cv_search.csv")

    print(search.round(4).to_string())
    print("challenger (raw):", {k: round(v, 4) for k, v in metrics.items()})
    return model


def calibrate(name: str, p_raw: np.ndarray, y: np.ndarray) -> tuple[Calibrator, np.ndarray]:
    """Compare sigmoid and isotonic out-of-fold on the calibration set, keep the lower Brier,
    refit it on the whole calibration set. Returns the calibrator and the out-of-fold PDs."""
    table = compare_calibrations(p_raw, y)
    method = table.loc[["sigmoid", "isotonic"], "brier"].idxmin()
    calibrator = Calibrator(method).fit(p_raw, y)
    oof = oof_calibrate(p_raw, y, method)

    curves = {
        "raw": reliability_table(y, p_raw),
        f"calibrated ({method})": reliability_table(y, oof),
    }
    reliability_plot(
        curves,
        f"{name}: reliability before and after calibration",
        FIGURES / f"reliability_{name}.png",
    )
    with mlflow.start_run(run_name=f"calibration_{name}"):
        mlflow.log_param("method", method)
        mlflow.log_metrics({f"brier_{k}": v for k, v in table["brier"].items()})
    print(f"{name} calibration:\n{table.round(5)}")
    return calibrator, oof


def choose_threshold(y: np.ndarray, pd_hat: np.ndarray, ead: np.ndarray, name: str) -> dict:
    """Cost-minimising threshold on the calibration set (out-of-fold calibrated PDs)."""
    costs = CONFIG["costs"]
    curve = cost_curve(y, pd_hat, ead, costs["lgd"], costs["margin"], np.linspace(0.005, 0.5, 200))
    curve["cost_per_1000"] = curve["cost"] / len(y) * 1000
    best = optimal_threshold(curve)
    breakeven = breakeven_threshold(costs["lgd"], costs["margin"])
    sens = sensitivity(y, pd_hat, ead, costs["lgd_grid"], [0.03, costs["margin"], 0.08])
    if name == "scorecard":
        cost_plot(
            curve,
            best,
            breakeven,
            "Expected cost is lowest around the break-even PD",
            FIGURES / "cost_vs_threshold.png",
        )
    return {"threshold": best, "breakeven": breakeven, "sensitivity": sens.to_dict("records")}


def main() -> None:
    setup_mlflow()
    MODELS.mkdir(exist_ok=True)
    train, calib, _ = load_splits()  # the test set stays untouched
    X_calib, y_calib = xy(calib)
    ead = calib[CONFIG["costs"]["ead_column"]].to_numpy()

    models = {
        "scorecard": train_scorecard(train, calib),
        "challenger": train_challenger(train, calib),
    }
    calibrators, decision = {}, {"lgd": CONFIG["costs"]["lgd"], "margin": CONFIG["costs"]["margin"]}
    for name, model in models.items():
        calibrators[name], oof = calibrate(name, model.predict_pd(X_calib), y_calib.to_numpy())
        decision[name] = choose_threshold(y_calib.to_numpy(), oof, ead, name)
        print(f"{name} threshold: {decision[name]['threshold']:.4f}")

    joblib.dump(models["scorecard"], MODELS / "scorecard.joblib")
    joblib.dump(models["challenger"], MODELS / "challenger.joblib")
    joblib.dump(calibrators, MODELS / "calibrators.joblib")
    (MODELS / "decision.json").write_text(json.dumps(decision, indent=2))
    with mlflow.start_run(run_name="decision"):
        mlflow.log_params({"lgd": decision["lgd"], "margin": decision["margin"]})
        mlflow.log_metrics({f"threshold_{n}": decision[n]["threshold"] for n in models})
        mlflow.log_artifacts(str(MODELS), artifact_path="models")


if __name__ == "__main__":
    main()
