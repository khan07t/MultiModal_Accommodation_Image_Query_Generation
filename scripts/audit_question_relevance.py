"""Measure cross-amenity drift in the generated questions.

    python scripts/audit_question_relevance.py

Fallback rate and well-formedness say nothing about whether a question is about
the amenity that was actually detected. This script measures that gap directly:
for each amenity, what fraction of its questions reference a *different* amenity?

Matching is stem-based rather than exact-word, because the obvious failures do
not all use the noun. "Can I easily reach everything I need while bathing
alone?" is a bathroom question attached to a kettle detection, and a naive
search for "bath" as a whole word misses it.

The numbers this prints are the ones quoted in the README, so the claim stays
checkable rather than asserted.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Stems, not whole words: "bath" catches bathing, bathe, bathtub, bathroom.
AMENITY_STEMS = {
    "bathtub": ["bath", "tub", "shower", "soak"],
    "kettle": ["kettle", "boil", "tea", "coffee", "brew"],
    "hairdryer": ["hairdry", "hair dry", "blow dry", "blow-dry", "dryer"],
    "mirror": ["mirror", "reflect", "vanity"],
    "tv": ["tv", "televis", "screen", "stream", "channel"],
}


def drift_mask(questions: list[str], own: str) -> list[bool]:
    """True where a question mentions another amenity but not its own."""
    others = [s for key, stems in AMENITY_STEMS.items() if key != own for s in stems]
    own_pattern = re.compile("|".join(re.escape(s) for s in AMENITY_STEMS[own]), re.I)
    other_pattern = re.compile("|".join(re.escape(s) for s in others), re.I)
    return [
        bool(other_pattern.search(q)) and not bool(own_pattern.search(q))
        for q in questions
    ]


def main() -> int:
    rows = []
    for path in sorted(glob.glob(str(
        REPO_ROOT / "results" / "question_generation_outputs" / "*" / "*_gemini3_perspectives.csv"
    ))):
        amenity = Path(path).parent.name
        frame = pd.read_csv(path)
        cols = [c for c in frame.columns if c.startswith("gemini_3_")]
        questions = [str(v).strip() for c in cols for v in frame[c].dropna() if str(v).strip()]
        flags = drift_mask(questions, amenity)
        drifted = sum(flags)
        rows.append({
            "amenity": amenity,
            "questions": len(questions),
            "off-topic": drifted,
            "rate": f"{drifted / len(questions):.1%}" if questions else "n/a",
        })
        if amenity == "kettle":
            examples = [q for q, f in zip(questions, flags) if f][:3]

    table = pd.DataFrame(rows)
    total_q = table["questions"].sum()
    total_d = table["off-topic"].sum()
    print(table.to_markdown(index=False))
    print(f"\nOverall: {total_d} of {total_q} questions mention another amenity "
          f"and not their own ({total_d / total_q:.1%}).")
    if examples:
        print("\nExample drift on kettle detections:")
        for q in examples:
            print(f"  - {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
