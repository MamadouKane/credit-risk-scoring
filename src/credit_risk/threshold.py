"""Decision: approve when PD < threshold, threshold set by money, not by 0.5.

Cost of an approved applicant who defaults: LGD x EAD (the bank loses part of the loan).
Cost of a rejected applicant who would have repaid: margin x EAD (the profit we gave up).
EAD (exposure at default) is approximated by AMT_CREDIT. LGD and margin are assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_cost(
    y: np.ndarray, pd_hat: np.ndarray, ead: np.ndarray, threshold: float, lgd: float, margin: float
) -> float:
    """Money lost on a portfolio when approving every applicant with PD < threshold."""
    y, pd_hat, ead = np.asarray(y), np.asarray(pd_hat), np.asarray(ead, dtype=float)
    approved = pd_hat < threshold
    loss_on_bads = lgd * ead[approved & (y == 1)].sum()
    lost_margin = margin * ead[~approved & (y == 0)].sum()
    return float(loss_on_bads + lost_margin)


def cost_curve(
    y: np.ndarray,
    pd_hat: np.ndarray,
    ead: np.ndarray,
    lgd: float,
    margin: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Cost and approval rate for a grid of thresholds (default: 0.5 % to 50 %)."""
    thresholds = np.linspace(0.005, 0.5, 100) if thresholds is None else thresholds
    pd_hat = np.asarray(pd_hat)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "cost": [expected_cost(y, pd_hat, ead, t, lgd, margin) for t in thresholds],
            "approval_rate": [(pd_hat < t).mean() for t in thresholds],
        }
    )


def optimal_threshold(curve: pd.DataFrame) -> float:
    """Threshold with the lowest cost on the curve."""
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


def breakeven_threshold(lgd: float, margin: float) -> float:
    """Theory for calibrated PDs: approve if the expected loss is below the expected margin,
    PD x LGD < (1 - PD) x margin, i.e. PD < margin / (LGD + margin)."""
    return margin / (lgd + margin)


def sensitivity(
    y: np.ndarray,
    pd_hat: np.ndarray,
    ead: np.ndarray,
    lgd_grid: list[float],
    margin_grid: list[float],
) -> pd.DataFrame:
    """Optimal threshold and approval rate per (LGD, margin) pair: what a risk manager asks."""
    rows = []
    for lgd in lgd_grid:
        for margin in margin_grid:
            curve = cost_curve(y, pd_hat, ead, lgd, margin, np.linspace(0.005, 0.5, 200))
            best = optimal_threshold(curve)
            rows.append(
                {
                    "lgd": lgd,
                    "margin": margin,
                    "optimal_threshold": best,
                    "breakeven": breakeven_threshold(lgd, margin),
                    "approval_rate": float((np.asarray(pd_hat) < best).mean()),
                }
            )
    return pd.DataFrame(rows)


def decision_summary(
    y: np.ndarray, pd_hat: np.ndarray, ead: np.ndarray, threshold: float, lgd: float, margin: float
) -> dict[str, float]:
    """The numbers a credit committee reads: approval, default among approved, money saved."""
    y, pd_hat = np.asarray(y), np.asarray(pd_hat)
    approved = pd_hat < threshold
    cost = expected_cost(y, pd_hat, ead, threshold, lgd, margin)
    cost_all = expected_cost(y, pd_hat, ead, np.inf, lgd, margin)  # approve everyone
    n = len(y)
    return {
        "threshold": float(threshold),
        "approval_rate": float(approved.mean()),
        "default_rate_approved": float(y[approved].mean()) if approved.any() else 0.0,
        "cost": cost,
        "cost_approve_all": cost_all,
        "cost_per_1000": cost / n * 1000,
        "cost_approve_all_per_1000": cost_all / n * 1000,
        "saving_per_1000": (cost_all - cost) / n * 1000,
    }


def expected_loss_by_grade(
    grades: pd.Series, pd_hat: np.ndarray, ead: np.ndarray, y: np.ndarray, lgd: float
) -> pd.DataFrame:
    """Illustrative credit-risk view: EL = PD x LGD x EAD per grade, with an ASSUMED LGD.

    A regulatory setup would use a 12-month PD on the 90-days-past-due definition, LGD and EAD
    models of their own, and IFRS 9 staging (lifetime PD for loans whose risk has increased).
    """
    frame = pd.DataFrame(
        {
            "grade": np.asarray(grades),
            "pd": np.asarray(pd_hat),
            "ead": np.asarray(ead, dtype=float),
            "y": np.asarray(y),
        }
    )
    frame["el"] = frame["pd"] * lgd * frame["ead"]
    out = frame.groupby("grade").agg(
        count=("y", "size"),
        mean_pd=("pd", "mean"),
        observed_default_rate=("y", "mean"),
        exposure=("ead", "sum"),
        expected_loss=("el", "sum"),
    )
    out["el_rate"] = out["expected_loss"] / out["exposure"]
    return out
