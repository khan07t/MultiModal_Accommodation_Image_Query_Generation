"""Generate amenity questions from a detection CSV.

Examples
--------
    python -m src.question_generation.generate \
        --detections results/detection_outputs/kettle/dets_kettle_baseline.csv \
        --backend flan-t5-base \
        --out results/question_generation_outputs/kettle/questions_flan_base.csv

    python -m src.question_generation.generate \
        --detections results/detection_outputs/tv/dets_tv_baseline.csv \
        --backend gemini-flash --profile family --limit 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.question_generation.prompts import build_prompt, detection_summary  # noqa: E402

LOGGER = logging.getLogger("question_generation")


def prepare_prompts(detections: pd.DataFrame, profile: str) -> pd.DataFrame:
    frame = detections.copy()
    if "source_prompt" not in frame.columns:
        frame["source_prompt"] = "baseline"
    frame["detection_summary"] = frame.apply(detection_summary, axis=1)
    frame["llm_prompt"] = frame["detection_summary"].apply(
        lambda summary: build_prompt(summary, profile=profile)
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", default="generic")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and write them without calling any model")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    detections = pd.read_csv(args.detections)
    detections = detections[detections["score"] >= args.min_score]
    if args.limit:
        detections = detections.head(args.limit)
    LOGGER.info("Loaded %d detections from %s", len(detections), args.detections.name)

    frame = prepare_prompts(detections, args.profile)

    if args.dry_run:
        LOGGER.info("Dry run: skipping model call")
        frame["generated_question"] = ""
    else:
        from src.question_generation.backends import get_backend

        backend = get_backend(args.backend)
        LOGGER.info("Generating with %s", backend.name)
        frame["generated_question"] = backend.generate(frame["llm_prompt"].tolist())

    frame["backend"] = args.backend
    frame["profile"] = args.profile

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    LOGGER.info("Wrote %d rows to %s", len(frame), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
