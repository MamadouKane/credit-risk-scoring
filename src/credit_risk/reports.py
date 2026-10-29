"""Figures for the README and the model card. One message per chart, the title states it."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

BLUE, ORANGE, GREY, RED = "#2563eb", "#d97706", "#6b7280", "#dc2626"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def reliability_plot(curves: dict[str, pd.DataFrame], title: str, path: Path) -> Path:
    """Mean predicted PD vs observed default rate per decile, one line per version."""
    fig, ax = plt.subplots(figsize=(6, 5))
    top = max(max(t["mean_predicted"].max(), t["observed_rate"].max()) for t in curves.values())
    ax.plot([0, top], [0, top], color=GREY, ls="--", lw=1, label="perfect calibration")
    for (name, table), color in zip(curves.items(), [ORANGE, BLUE, RED], strict=False):
        ax.plot(table["mean_predicted"], table["observed_rate"], "o-", color=color, label=name)
    ax.set_xlabel("mean predicted PD (decile)")
    ax.set_ylabel("observed default rate")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    return _save(fig, path)


def cost_plot(curve: pd.DataFrame, best: float, breakeven: float, title: str, path: Path) -> Path:
    """Expected cost per 1,000 applications as a function of the threshold."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["threshold"], curve["cost_per_1000"] / 1e6, color=BLUE)
    ax.axvline(best, color=RED, lw=1.5, label=f"optimal threshold {best:.3f}")
    ax.axvline(breakeven, color=GREY, ls="--", lw=1, label=f"break-even PD {breakeven:.3f}")
    ax.set_xlabel("PD threshold (approve below)")
    ax.set_ylabel("expected cost per 1,000 applications (millions)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    return _save(fig, path)


def grade_plot(table: pd.DataFrame, title: str, path: Path) -> Path:
    """Observed default rate per grade, with counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table.index, table["observed_default_rate"], color=BLUE)
    for i, (rate, n) in enumerate(zip(table["observed_default_rate"], table["count"], strict=True)):
        ax.text(i, rate, f"{rate:.1%}\n{n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("observed default rate (test set)")
    ax.set_ylim(0, table["observed_default_rate"].max() * 1.25)
    ax.set_title(title, fontsize=10)
    return _save(fig, path)


def roc_plot(y: np.ndarray, models: dict[str, np.ndarray], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    for (name, p), color in zip(models.items(), [BLUE, ORANGE], strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, label=name)
    ax.set_xlabel("share of goods rejected (FPR)")
    ax.set_ylabel("share of bads caught (TPR)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    return _save(fig, path)


def psi_plot(report: pd.DataFrame, title: str, path: Path) -> Path:
    report = report.sort_values("psi")
    colors = [RED if v >= 0.25 else ORANGE if v >= 0.1 else BLUE for v in report["psi"]]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(report))))
    ax.barh(report["variable"], report["psi"], color=colors)
    ax.axvline(0.1, color=GREY, ls="--", lw=1)
    ax.axvline(0.25, color=RED, ls="--", lw=1)
    ax.set_xlabel("PSI")
    ax.set_title(title, fontsize=10)
    return _save(fig, path)
