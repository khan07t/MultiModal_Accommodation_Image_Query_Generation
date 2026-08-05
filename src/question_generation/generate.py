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

Vision backends need the source images as well as the detection rows, because
they are conditioned on the cropped region rather than a text summary of it:

    python -m src.question_generation.generate \
        --detections results/detection_outputs/bathtub/dets_bathtub_baseline.csv \
        --backend gemini-vision --image-root data/sample/bathtub \
        --perspective couple \
        --out /tmp/questions.csv
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


def generate_vision(frame: pd.DataFrame, backend, image_root: Path,
                    perspective: str, num_questions: int) -> List[str]:
    """Run a vision backend over each detection, one crop at a time.

    Vision backends cannot use the text prompt built above, so this path passes
    the image and box instead. Missing image files are recorded rather than
    raised, so one absent file does not abandon a long run.
    """
    from src.question_generation.backends import FALLBACK_MARKER

    answers: List[str] = []
    missing = 0
    for row in frame.itertuples():
        path = image_root / row.image
        if not path.exists():
            missing += 1
            answers.append(f"{FALLBACK_MARKER} FileNotFound")
            continue
        questions = backend.generate_for_detection(
            path,
            box=[row.x1, row.y1, row.x2, row.y2],
            label=row.label,
            num_questions=num_questions,
            perspective=perspective,
        )
        answers.append(" | ".join(questions))

    if missing:
        LOGGER.warning("%d of %d images not found under %s",
                       missing, len(frame), image_root)
    return answers


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
    parser.add_argument("--image-root", type=Path, default=None,
                        help="Directory holding the source images. Required for "
                             "vision backends, ignored by text-only ones.")
    parser.add_argument("--perspective", default="single",
                        help="Travelling party for vision backends: single, couple or group")
    parser.add_argument("--num-questions", type=int, default=3,
                        help="Questions per detection, vision backends only")
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

    from src.question_generation.backends import VISION_BACKENDS

    is_vision = args.backend in VISION_BACKENDS

    if is_vision and args.image_root is None:
        LOGGER.error(
            "--backend %s is conditioned on the cropped image, so it needs "
            "--image-root pointing at the directory holding those images, "
            "e.g. --image-root data/sample/bathtub",
            args.backend,
        )
        return 1

    if args.dry_run:
        LOGGER.info("Dry run: skipping model call")
        frame["generated_question"] = ""
    else:
        from src.question_generation.backends import get_backend

        backend = get_backend(args.backend)
        LOGGER.info("Generating with %s", backend.name)
        if is_vision:
            frame["generated_question"] = generate_vision(
                frame, backend, args.image_root, args.perspective, args.num_questions
            )
        else:
            frame["generated_question"] = backend.generate(frame["llm_prompt"].tolist())

    frame["backend"] = args.backend
    frame["profile"] = args.perspective if is_vision else args.profile

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    LOGGER.info("Wrote %d rows to %s", len(frame), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
