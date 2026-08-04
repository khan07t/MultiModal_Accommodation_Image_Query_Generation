"""Figure generation for the detection experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 140


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return path


def plot_f1_vs_threshold(summary: pd.DataFrame, display_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for prompt, group in summary.groupby("prompt"):
        group = group.sort_values("conf")
        ax.plot(group["conf"], group["f1"], marker="o", label=prompt)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1")
    ax.set_title(f"F1 vs confidence threshold: {display_name}")
    ax.grid(alpha=0.3)
    ax.legend(title="Prompt set")
    return _save(fig, out_dir / f"F1_vs_threshold_{display_name}.png")


def plot_prompt_comparison(summary: pd.DataFrame, display_name: str, out_dir: Path) -> Path:
    best = (
        summary.sort_values(["prompt", "f1"], ascending=[True, False])
        .groupby("prompt", as_index=False)
        .head(1)
        .sort_values("f1", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(best))
    width = 0.27
    ax.bar([i - width for i in x], best["precision"], width, label="Precision")
    ax.bar(list(x), best["recall"], width, label="Recall")
    ax.bar([i + width for i in x], best["f1"], width, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(best["prompt"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Best score per prompt set: {display_name}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return _save(fig, out_dir / "prompt_comparison_full.png")


def plot_negative_curve(negatives: pd.DataFrame, display_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(negatives["conf"], negatives["specificity"], marker="o", label="Specificity (1−FPR)")
    ax.plot(negatives["conf"], negatives["fpr"], marker="x", label="FPR")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Rate")
    ax.set_title(f"Negative-set behaviour: {display_name}")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, out_dir / f"neg_curve_{display_name.lower()}.png")


def plot_precision_recall(summary: pd.DataFrame, prompt: str, display_name: str, out_dir: Path) -> Path:
    group = summary[summary["prompt"] == prompt].sort_values("conf")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(group["recall"], group["precision"], marker="o")
    for _, row in group.iterrows():
        ax.annotate(f"{row['conf']:.2f}", (row["recall"], row["precision"]), fontsize=7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR: {display_name} ({prompt})")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir / f"PR_{display_name.lower()}_{prompt}.png")
