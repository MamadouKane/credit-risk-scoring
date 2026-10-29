"""Final evaluation on the TEST set, run once after training (make evaluate).

Writes docs/results.json (every number quoted in the README) and the result figures.
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from credit_risk.config import ROOT, load_config, project_path  # noqa: E402
from credit_risk.data import clean, load_table  # noqa: E402
from credit_risk.drift import drift_report, shifted_batch  # noqa: E402
from credit_risk.explain import fairness_table, reason_codes, shap_contributions  # noqa: E402
from credit_risk.features import build_features  # noqa: E402
from credit_risk.metrics import summary  # noqa: E402
from credit_risk.reports import grade_plot, psi_plot, roc_plot  # noqa: E402
from credit_risk.threshold import decision_summary, expected_loss_by_grade  # noqa: E402
from credit_risk.train import load_splits, xy  # noqa: E402

CONFIG = load_config()
MODELS, FIGURES = project_path("models"), project_path("figures")


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    if isinstance(obj, float | np.floating):
        return round(float(obj), 4)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def shap_figures(
    challenger, X: pd.DataFrame, approved_row: pd.DataFrame, rejected_row: pd.DataFrame
) -> None:
    """Global beeswarm on a sample, and the waterfall of one rejected applicant."""
    sample = X.sample(5_000, random_state=0)
    contrib = shap_contributions(challenger, sample)
    numeric = sample.apply(lambda c: c.cat.codes if hasattr(c, "cat") else c)
    data = challenger.prepared(sample).apply(lambda c: c.cat.codes if hasattr(c, "cat") else c)
    explanation = shap.Explanation(
        values=contrib.drop(columns="BIAS").to_numpy(),
        base_values=contrib["BIAS"].to_numpy(),
        data=data.to_numpy(dtype=float),
        feature_names=list(numeric.columns),
    )
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("XGBoost: what pushes the PD up (right) or down (left)", fontsize=10)
    plt.savefig(FIGURES / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    for name, row in (("rejected", rejected_row), ("approved", approved_row)):
        one = shap_contributions(challenger, row)
        exp = shap.Explanation(
            values=one.drop(columns="BIAS").to_numpy()[0],
            base_values=float(one["BIAS"].iloc[0]),
            data=row.iloc[0].to_numpy(),
            feature_names=list(row.columns),
        )
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.title(f"One {name} applicant: contributions to the log-odds of default", fontsize=10)
        plt.savefig(FIGURES / f"shap_waterfall_{name}.png", dpi=150, bbox_inches="tight")
        plt.close("all")


def main() -> None:
    scorecard = joblib.load(MODELS / "scorecard.joblib")
    challenger = joblib.load(MODELS / "challenger.joblib")
    calibrators = joblib.load(MODELS / "calibrators.joblib")
    decision = json.loads((MODELS / "decision.json").read_text())
    lgd, margin = decision["lgd"], decision["margin"]

    train, _, test = load_splits()
    X_test, y_test = xy(test)
    y = y_test.to_numpy()
    ead = test[CONFIG["costs"]["ead_column"]].to_numpy()

    pd_sc = calibrators["scorecard"].predict(scorecard.predict_pd(X_test))
    pd_ch = calibrators["challenger"].predict(challenger.predict_pd(X_test))
    results: dict = {"n_test": len(test), "default_rate_test": float(y.mean())}
    results["metrics"] = {"scorecard": summary(y, pd_sc), "challenger": summary(y, pd_ch)}
    results["gini_gain"] = (
        results["metrics"]["challenger"]["gini"] - results["metrics"]["scorecard"]["gini"]
    )

    results["decision"] = {
        "scorecard": decision_summary(
            y, pd_sc, ead, decision["scorecard"]["threshold"], lgd, margin
        ),
        "challenger": decision_summary(
            y, pd_ch, ead, decision["challenger"]["threshold"], lgd, margin
        ),
    }
    results["sensitivity"] = decision["scorecard"]["sensitivity"]

    grades = scorecard.grade(X_test)
    el = expected_loss_by_grade(grades, pd_sc, ead, y, lgd)
    results["grades"] = el.reset_index().to_dict("records")
    grade_plot(
        el, "Each grade defaults more than the one above it (test set)", FIGURES / "grades_test.png"
    )
    roc_plot(
        y,
        {"scorecard": pd_sc, "XGBoost challenger": pd_ch},
        "ROC curves on the test set",
        FIGURES / "roc_test.png",
    )

    approved = pd_sc < decision["scorecard"]["threshold"]
    age_band = pd.cut(test["AGE_YEARS"], [20, 25, 35, 45, 55, 70])
    results["fairness"] = {
        "gender": fairness_table(test["CODE_GENDER"].fillna("unknown"), approved, y)
        .reset_index()
        .to_dict("records"),
        "age_band": fairness_table(age_band.astype(str), approved, y)
        .reset_index()
        .to_dict("records"),
    }

    rejected_idx = int(np.argmax(pd_sc))  # the riskiest test applicant
    approved_idx = int(np.argmin(pd_sc))
    rejected = X_test.iloc[[rejected_idx]]
    results["example_rejected"] = {
        "pd": float(pd_sc[rejected_idx]),
        "score": float(scorecard.score(rejected)[0]),
        "grade": str(scorecard.grade(rejected)[0]),
        "reasons": reason_codes(scorecard, rejected)[0],
    }
    shap_figures(challenger, X_test, X_test.iloc[[approved_idx]], rejected)

    # Drift: train vs the Kaggle test population, and vs a simulated shifted batch
    variables = [*scorecard.features, "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL"]
    kaggle = build_features(
        clean(load_table("application_test")),
        load_table("bureau"),
        load_table("installments_payments"),
    )
    real = drift_report(train, kaggle, variables)
    simulated = drift_report(
        train, shifted_batch(train, {"AMT_INCOME_TOTAL": 1.3, "EXT_SOURCE_3": 0.8}), variables
    )
    psi_plot(
        real,
        "Train vs Kaggle test population: only the product mix moves",
        FIGURES / "psi_train_vs_kaggle_test.png",
    )
    psi_plot(
        simulated,
        "Simulated batch (income +30 %, external score -20 %): PSI catches it",
        FIGURES / "psi_simulated_shift.png",
    )
    results["drift"] = {
        "kaggle_test": real.to_dict("records"),
        "simulated": simulated.to_dict("records"),
    }

    scorecard_table = []
    for name in scorecard.features:
        table = (
            scorecard.binning.get_binned_variable(name).binning_table.build().drop(index="Totals")
        )
        table = table[table["Count"].astype(float) > 0]
        scorecard_table.append(
            pd.DataFrame(
                {
                    "variable": name,
                    "bin": table["Bin"].astype(str),
                    "count": table["Count"],
                    "event_rate": table["Event rate"],
                    "woe": table["WoE"].astype(float),
                }
            )
        )
    pd.concat(scorecard_table).to_csv(ROOT / "docs" / "scorecard_bins.csv", index=False)

    (ROOT / "docs" / "results.json").write_text(json.dumps(_round(results), indent=2, default=str))
    print(
        json.dumps(_round({k: results[k] for k in ("metrics", "gini_gain", "decision")}), indent=2)
    )


if __name__ == "__main__":
    main()
