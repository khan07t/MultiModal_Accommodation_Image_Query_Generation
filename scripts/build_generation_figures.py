"""Render the question-generation figures from committed CSVs.

    python scripts/build_generation_figures.py

The detection stage already had figures (PR curves, threshold sweeps). The
generation stage did not, so these are built here from the committed analysis
CSVs. Every value plotted is regenerated from a file in `results/`.

Outputs to results/figures/generation/:
    model_selection.png      BERTScore + fallback rate across three Gemini variants
    generation_quality.png   semantic diversity and fallback rate per amenity
    profile_conditioning.png distinctiveness between traveller-profile pairs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "results" / "analysis_summary"
OUT_DIR = REPO_ROOT / "results" / "figures" / "generation"

INK = "#1F2933"
MUTED = "#5B6B7B"
BLUE = "#2F5D8C"
ORANGE = "#C2703D"
GREEN = "#2E7D52"
GREY = "#B4BCC4"
LABELS = {"bathtub": "Bathtub", "tv": "TV", "kettle": "Kettle",
          "hairdryer": "Hairdryer", "mirror": "Mirror"}


def style(ax, title, subtitle=None):
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=14)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.8,
                color=MUTED, va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GREY, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)


def model_selection():
    bert = pd.read_csv(SUMMARY / "model_comparison" / "bertscore_by_amenity.csv")
    fallback = pd.read_csv(SUMMARY / "model_comparison" / "model_fallback_rates_summary.csv")

    overall = bert.groupby("model")["bertscore"].mean().sort_values()
    fb = fallback.set_index("Model")["Fallback Rate (%)"]
    # CSV shortens the winning model's name; match it back to the BERTScore table
    fb.index = [i.replace("Gemini 3.0 Pro", "Gemini 3.0 Pro Preview") for i in fb.index]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    colors = [GREEN if "3.0" in m else BLUE for m in overall.index]
    ax.barh(range(len(overall)), overall.values, color=colors, height=0.55)
    ax.set_yticks(range(len(overall)))
    ax.set_yticklabels(overall.index, fontsize=9)
    ax.set_xlim(0.66, 0.72)
    for i, v in enumerate(overall.values):
        ax.text(v + 0.0008, i, f"{v:.3f}", va="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.grid(axis="x", color=GREY, alpha=0.35, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    style(ax, "Semantic similarity to reference questions",
          "mean BERTScore across five amenities, differences under 2%")
    ax.set_xlabel("BERTScore", fontsize=9, color=MUTED)

    ax = axes[1]
    ordered = fb.reindex(overall.index)
    colors = [GREEN if "3.0" in m else BLUE for m in ordered.index]
    ax.barh(range(len(ordered)), ordered.values, color=colors, height=0.55)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(["" for _ in ordered.index])
    for i, v in enumerate(ordered.values):
        ax.text(v + 0.12, i, f"{v:.2f}%", va="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_xlim(0, 12)
    ax.grid(axis="x", color=GREY, alpha=0.35, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    style(ax, "Fallback rate", "share of 7,770 questions that failed to generate, lower is better")
    ax.set_xlabel("fallback rate (%)", fontsize=9, color=MUTED)

    fig.suptitle("Model selection: three Gemini variants on identical crops and prompts",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.008, ha="left", y=1.04)
    fig.text(0.008, 0.975,
             "Gemini 3.0 Pro Preview wins on both and was used for all subsequent experiments.",
             fontsize=9.2, color=MUTED, ha="left")
    plt.tight_layout()
    path = OUT_DIR / "model_selection.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    return path


def generation_quality():
    diversity = pd.read_csv(SUMMARY / "semantic_diversity_by_amenity.csv")
    fallback = pd.read_csv(SUMMARY / "model_comparison" / "gemini30_fallback_rates.csv")

    div = diversity[diversity["amenity"] != "overall"].copy()
    overall = float(diversity.loc[diversity["amenity"] == "overall", "mean_semantic_diversity"].iloc[0])
    div["label"] = div["amenity"].map(LABELS)
    div = div.sort_values("mean_semantic_diversity", ascending=False)

    fb = fallback[fallback["Amenity"] != "TOTAL"].copy()
    fb["amenity"] = fb["Amenity"].str.lower()
    fb["label"] = fb["amenity"].map(LABELS)
    fb = fb.set_index("label").reindex(div["label"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.bar(div["label"], div["mean_semantic_diversity"], color=BLUE, width=0.6)
    ax.axhline(overall, color=ORANGE, linestyle="--", linewidth=1.4)
    ax.text(len(div) - 0.4, overall + 0.012, f"overall {overall:.3f}",
            color=ORANGE, fontsize=8.6, ha="right", fontweight="bold")
    for i, v in enumerate(div["mean_semantic_diversity"]):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.8, color=INK)
    ax.set_ylim(0, 0.95)
    style(ax, "Semantic diversity within each image's question set",
          "1 − mean pairwise cosine similarity; higher means less redundant")

    ax = axes[1]
    colors = [ORANGE if v > 8 else BLUE for v in fb["Fallback Rate (%)"]]
    ax.bar(fb.index, fb["Fallback Rate (%)"], color=colors, width=0.6)
    for i, v in enumerate(fb["Fallback Rate (%)"]):
        ax.text(i, v + 0.25, f"{v:.2f}%", ha="center", fontsize=8.8, color=INK)
    ax.set_ylim(0, 13)
    style(ax, "Fallback rate by amenity",
          "generation failures cluster on visually variable amenities")

    fig.suptitle("Question quality: Gemini 3.0 Pro Preview",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.008, ha="left", y=1.04)
    fig.text(0.008, 0.975,
             "Mirror is worst on both. TV is the counter-example: the most reliable amenity "
             "sits only mid-table on diversity, so the two are not interchangeable.",
             fontsize=9.2, color=MUTED, ha="left")
    plt.tight_layout()
    path = OUT_DIR / "generation_quality.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    return path


def profile_conditioning():
    dist = pd.read_csv(SUMMARY / "profile_distinctiveness.csv")
    pivot = dist.pivot(index="amenity", columns="pair", values="distinctiveness")
    pivot.index = [LABELS.get(i, i) for i in pivot.index]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    pretty = {"single_vs_couple": "single vs couple",
              "single_vs_group": "single vs group",
              "couple_vs_group": "couple vs group"}
    colors = {"single_vs_couple": BLUE, "single_vs_group": GREEN, "couple_vs_group": ORANGE}

    width, cols = 0.26, list(pivot.columns)
    for offset, col in enumerate(cols):
        positions = [i + (offset - 1) * width for i in range(len(pivot))]
        ax.bar(positions, pivot[col], width=width,
               label=pretty.get(col, col), color=colors.get(col, BLUE))

    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=8.8, ncol=3, loc="upper right")
    style(ax, "Does conditioning on the travelling party change the questions?",
          "1 − cosine similarity between question sets generated for different profiles")

    fig.suptitle("Traveller-profile conditioning",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.008, ha="left", y=1.03)
    fig.text(0.008, 0.972,
             "Every pair separates well above zero, so the model attends to the profile "
             "rather than ignoring it. Single vs group separates most.",
             fontsize=9.2, color=MUTED, ha="left")
    plt.tight_layout()
    path = OUT_DIR / "profile_conditioning.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (model_selection, generation_quality, profile_conditioning):
        path = fn()
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
