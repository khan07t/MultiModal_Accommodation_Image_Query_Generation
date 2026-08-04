"""Regenerate the Finding 3 diversity table in README.md from committed CSVs.

Finding 3 is the claim that text-only conditioning collapses into templates and
that conditioning on the cropped image region does not. This script recomputes
every number in that table so the claim stays checkable:

    python scripts/build_finding3_table.py

Text-only questions come from the four `generated_question_*` columns of
results/question_generation_outputs/bathtub/merged_detections_bathtub_with_questions_all_models.csv
(1,003 rows). Vision questions come from the `gemini_3_*` columns of
results/question_generation_outputs/bathtub/dets_bathtub_baseline_gemini3_perspectives.csv
(276 detections x 3 perspectives x 3 questions = 2,484).

distinct-n falls as a corpus grows, so comparing 2,484 vision questions against
1,003 text-only ones would flatter the vision numbers. The vision row is
therefore reported as the mean over `--draws` random subsamples of exactly 1,003
questions, with the spread printed underneath.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import question_metrics as qm  # noqa: E402

BATHTUB = REPO_ROOT / "results" / "question_generation_outputs" / "bathtub"
TEXT_CSV = BATHTUB / "merged_detections_bathtub_with_questions_all_models.csv"
VISION_CSV = BATHTUB / "dets_bathtub_baseline_gemini3_perspectives.csv"

TEXT_BACKENDS = {
    "Flan-T5 Base": "generated_question_flan_base",
    "Flan-T5 Large": "generated_question_flan_large",
    "Llama 3 70B": "generated_question_llama70b",
    "Llama 3 8B": "generated_question_llama8b",
}


def clean(values) -> list[str]:
    return [str(v).strip() for v in values if str(v).strip() and str(v).lower() != "nan"]


def most_common(questions: list[str]) -> tuple[str, int]:
    if not questions:
        return ("n/a", 0)
    return Counter(questions).most_common(1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=20,
                        help="Subsamples used for the size-matched vision row")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for path in (TEXT_CSV, VISION_CSV):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    text_df = pd.read_csv(TEXT_CSV)
    n_reference = len(text_df)

    rows = []
    for display, column in TEXT_BACKENDS.items():
        if column not in text_df.columns:
            print(f"skip {display}: no column {column}", file=sys.stderr)
            continue
        questions = clean(text_df[column].dropna())
        summary = qm.summarize(questions)
        top, count = most_common(questions)
        rows.append({
            "Backend": display,
            "Conditioning": "text",
            "Distinct-2": f"{summary['distinct_2']:.3f}",
            "Duplicate rate": f"{summary['duplicate_rate']:.1%}",
            "Most common output": f"*“{top}”* × {count}",
        })

    vision_df = pd.read_csv(VISION_CSV)
    vision_cols = [c for c in vision_df.columns if c.startswith("gemini_3_")]
    vision = clean(v for c in vision_cols for v in vision_df[c].dropna())

    d2, dup = [], []
    for offset in range(args.draws):
        random.seed(args.seed + offset)
        sample = random.sample(vision, min(n_reference, len(vision)))
        summary = qm.summarize(sample)
        d2.append(summary["distinct_2"])
        dup.append(summary["duplicate_rate"])

    top, count = most_common(vision)
    rows.append({
        "Backend": "Gemini 3.0 Pro Preview",
        "Conditioning": "**vision**",
        "Distinct-2": f"**{statistics.mean(d2):.3f}**",
        "Duplicate rate": f"**{statistics.mean(dup):.1%}**",
        "Most common output": f"*“{top}”* × {count}",
    })

    print(pd.DataFrame(rows).to_markdown(index=False))
    print()
    print(f"Text-only rows: {n_reference}. Vision questions available: {len(vision)}, "
          f"subsampled to {min(n_reference, len(vision))} over {args.draws} draws.")
    print(f"Vision distinct-2 range {min(d2):.3f}–{max(d2):.3f}; "
          f"duplicate rate range {min(dup):.1%}–{max(dup):.1%}.")
    full = qm.summarize(vision)
    print(f"Vision on the full {len(vision)} questions: distinct-2 {full['distinct_2']:.3f}, "
          f"duplicate rate {full['duplicate_rate']:.1%}, "
          f"fallback {full['fallback_rate']:.1%}, "
          f"well-formed {full['well_formed_rate']:.1%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
