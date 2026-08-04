"""Run the OWL-ViT detection experiment for one or more amenities.

Replaces `object_detection_analysis_{bathtub,hairdryer,kettle,mirror,tv}.py`,
which were 86-99% identical to one another.

Examples
--------
    python -m src.detection.run_detection --amenity kettle --data-root data/
    python -m src.detection.run_detection --all --prompt-set baseline
    python -m src.detection.run_detection --amenity tv --skip-inference
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import AmenityConfig, load_config  # noqa: E402
from src.detection.metrics import (  # noqa: E402
    best_per_prompt,
    evaluate_negatives,
    sweep_thresholds,
)

# `src.detection.plots` pulls in matplotlib, and `detector` pulls in torch.
# Both are imported lazily, at the point of use, so that `--help` and
# `--skip-inference --no-plots` work with pandas alone. Reading the CLI help
# should not require a plotting library.

LOGGER = logging.getLogger("detection")
DEFAULT_OUT_ROOT = Path("results")


def negative_image_names(coco) -> List[str]:
    """Images in the test split with no ground-truth annotation for the target."""
    return [
        coco.loadImgs(img_id)[0]["file_name"]
        for img_id in coco.getImgIds()
        if not coco.getAnnIds(imgIds=img_id)
    ]


def is_same_directory(a: Path, b: Path) -> bool:
    """True if two paths name the same directory, however they are spelled.

    A plain `a == b` compares path *strings*, so `results` and
    `/abs/path/to/repo/results` compare unequal while pointing at the same
    place. Anything guarding against an overwrite has to resolve first, and
    normcase matters on Windows, where `Results` and `results` are the same
    directory but not the same string.
    """
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def resolve_image_dir(data_root: Path, slug: str) -> Path:
    """Locate the annotated image directory for one amenity.

    Two layouts are supported:

        <data-root>/<slug>/test/_annotations.coco.json   full Roboflow export
        <data-root>/<slug>/_annotations.coco.json        the committed sample

    The full datasets ship with train/valid/test subdirectories; `data/sample/`
    is a flat 46-image slice with no split level. Returning whichever exists
    means `--data-root data/sample` works on a clean checkout.
    """
    candidates = (data_root / slug / "test", data_root / slug)
    for candidate in candidates:
        if (candidate / "_annotations.coco.json").exists():
            return candidate
    # Nothing found: report the conventional location in the error.
    return candidates[0]


def run_amenity(
    cfg: AmenityConfig,
    data_root: Path,
    out_root: Path,
    prompt_sets: List[str] | None = None,
    skip_inference: bool = False,
    device: str | None = None,
    make_plots: bool = True,
) -> pd.DataFrame:
    """Run detection + evaluation for one amenity and write results to disk."""
    from pycocotools.coco import COCO

    test_dir = resolve_image_dir(data_root, cfg.dataset_slug)
    test_json = test_dir / "_annotations.coco.json"

    metrics_dir = out_root / "detection_metrics" / cfg.key
    detections_dir = out_root / "detection_outputs" / cfg.key
    figures_dir = out_root / "figures" / cfg.key

    # Only create what this run will actually write. `--skip-inference` reads
    # detections rather than producing them, and `--no-plots` writes no figures,
    # so creating those directories would leave empty folders behind.
    wanted = [metrics_dir]
    if not skip_inference:
        wanted.append(detections_dir)
    if make_plots:
        wanted.append(figures_dir)
    for directory in wanted:
        directory.mkdir(parents=True, exist_ok=True)

    # `--skip-inference` re-scores detections that are already committed under
    # results/. Those live in the repository, not in --out-root, so a run that
    # redirects its output still needs to read them from the default location.
    committed_detections = DEFAULT_OUT_ROOT / "detection_outputs" / cfg.key

    selected = prompt_sets or list(cfg.prompt_sets)
    detections_by_prompt: Dict[str, pd.DataFrame] = {}

    detector = None
    if not skip_inference:
        if not test_json.exists():
            raise FileNotFoundError(
                f"Missing COCO annotations at {test_json}. "
                "Download the datasets first (see docs/dataset_inventory.md), "
                "or pass --skip-inference to re-score committed detection CSVs."
            )
        from src.detection.detector import OwlViTDetector

        LOGGER.info("Loading %s", cfg.model_name)
        detector = OwlViTDetector(cfg.model_name, device=device)

    for prompt_set in selected:
        csv_path = detections_dir / f"dets_{cfg.key}_{prompt_set}.csv"

        if skip_inference:
            source = csv_path
            if not source.exists():
                source = committed_detections / f"dets_{cfg.key}_{prompt_set}.csv"
            if not source.exists():
                LOGGER.warning("No committed detections for %s, skipping", prompt_set)
                continue
            detections = pd.read_csv(source)
            LOGGER.info("Loaded %d detections from %s", len(detections), source)
        else:
            prompts = cfg.prompts(prompt_set)
            images = sorted(p for p in test_dir.glob("*.jpg"))
            detections = detector.detect_folder(
                images, prompts, threshold=cfg.inference_threshold
            )
            detections.to_csv(csv_path, index=False)
            LOGGER.info("Wrote %d detections to %s", len(detections), csv_path.name)

        detections_by_prompt[prompt_set] = detections

    if not detections_by_prompt:
        raise RuntimeError(f"No detections available for {cfg.key}")

    if not test_json.exists():
        LOGGER.warning(
            "No ground truth for %s at %s, so detections were written but not scored. "
            "Point --data-root at the full datasets, or at data/sample for the "
            "committed 46-image slice.",
            cfg.key,
            test_json,
        )
        return pd.DataFrame()

    coco = COCO(str(test_json))

    # Guard against a partial run silently overwriting the published results.
    # The committed detection CSVs cover the full test split; scoring them
    # against a smaller annotation set (data/sample) produces sample-scale
    # numbers that must not land in results/.
    scored_images = len(coco.getImgIds())
    detected_images = pd.concat(detections_by_prompt.values())["image"].nunique()
    sample_scale = scored_images < detected_images

    if sample_scale and is_same_directory(out_root, DEFAULT_OUT_ROOT):
        raise SystemExit(
            f"Refusing to overwrite {out_root}/.\n"
            f"  Ground truth covers {scored_images} images but the detections span "
            f"{detected_images}.\n"
            f"  Scoring a subset would replace the published results with "
            f"sample-scale numbers.\n"
            f"  Re-run with an explicit --out-root, e.g. --out-root /tmp/sample-run"
        )

    summary = sweep_thresholds(
        detections_by_prompt, coco, cfg.conf_sweep, cfg.eval_iou_threshold
    )
    summary_path = metrics_dir / f"summary_{cfg.key}.csv"
    summary.to_csv(summary_path, index=False)

    negatives = evaluate_negatives(
        pd.concat(detections_by_prompt.values(), ignore_index=True),
        negative_image_names(coco),
        cfg.neg_conf_sweep,
    )
    negatives.to_csv(metrics_dir / f"negatives_{cfg.key}.csv", index=False)

    if make_plots:
        from src.detection import plots

        plots.plot_f1_vs_threshold(summary, cfg.display_name, figures_dir)
        plots.plot_prompt_comparison(summary, cfg.display_name, figures_dir)
        plots.plot_negative_curve(negatives, cfg.display_name, figures_dir)

    best = best_per_prompt(summary).sort_values("f1", ascending=False)
    LOGGER.info(
        "%s best: prompt=%s conf=%.2f F1=%.3f%s",
        cfg.display_name,
        best.iloc[0]["prompt"],
        best.iloc[0]["conf"],
        best.iloc[0]["f1"],
        "   [SAMPLE-SCALE]" if sample_scale else "",
    )
    summary.attrs["sample_scale"] = sample_scale
    summary.attrs["scored_images"] = scored_images
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--amenity", help="Amenity key, e.g. kettle")
    target.add_argument("--all", action="store_true", help="Run every amenity")

    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--prompt-set",
        action="append",
        dest="prompt_sets",
        help="Restrict to one prompt set (repeatable)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help=("Re-score the committed detection CSVs instead of running the model. "
              "Needs ground-truth annotations under --data-root."),
    )
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    configs = load_config(args.config) if args.config else load_config()
    keys = list(configs) if args.all else [args.amenity]

    sample_scale_runs = []
    for key in keys:
        if key not in configs:
            LOGGER.error("Unknown amenity %r. Available: %s", key, ", ".join(configs))
            return 1
        summary = run_amenity(
            configs[key],
            data_root=args.data_root,
            out_root=args.out_root,
            prompt_sets=args.prompt_sets,
            skip_inference=args.skip_inference,
            device=args.device,
            make_plots=not args.no_plots,
        )
        if summary.attrs.get("sample_scale"):
            sample_scale_runs.append((key, summary.attrs.get("scored_images", 0)))

    if sample_scale_runs:
        total = sum(n for _, n in sample_scale_runs)
        print(
            "\n" + "=" * 72
            + f"\nSAMPLE-SCALE RUN. Scored {total} images across "
              f"{len(sample_scale_runs)} amenities.\n"
              "These numbers are NOT comparable to the table in README.md, which\n"
              "comes from the full test split (1,531 images). A handful of images\n"
              "per amenity produces unstable and usually flattering F1.\n"
              "To reproduce the published table: python scripts/build_results_table.py\n"
            + "=" * 72,
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
