"""Regenerate the headline results table in README.md from the committed CSVs.

Run this after re-running detection to keep the README honest:
    python scripts/build_results_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.detection.metrics import best_per_prompt, load_summary  # noqa: E402


def main() -> int:
    configs = load_config()
    rows = []

    for key, cfg in configs.items():
        path = REPO_ROOT / "results" / "detection_metrics" / key / f"summary_{key}.csv"
        if not path.exists():
            print(f"skip {key}: no summary at {path}", file=sys.stderr)
            continue
        summary = load_summary(path)
        best = best_per_prompt(summary).sort_values("f1", ascending=False).iloc[0]
        rows.append(
            {
                "Amenity": cfg.display_name,
                "Best prompt": best["prompt"],
                "Conf": f"{best['conf']:.2f}",
                "F1": f"{best['f1']:.3f}",
                "Precision": f"{best['precision']:.3f}",
                "Recall": f"{best['recall']:.3f}",
                "mIoU": f"{best['mean_iou']:.3f}",
            }
        )

    print(pd.DataFrame(rows).to_markdown(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
