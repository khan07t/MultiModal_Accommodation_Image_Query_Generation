"""Curate the small, committable sample in `data/sample/`.

The full test splits are ~1,500 images and are not committed. This script selects
a deterministic 10-image slice per amenity so the notebooks can run on real data,
and writes the attribution file that ships alongside it.

The images are accommodation listing photographs provided by trivago N.V. and
reproduced with permission. They are not covered by this repository's MIT
licence. See `data/sample/ATTRIBUTION.md` and `docs/dataset_inventory.md`.

Selection, per amenity:
  * 8 images carrying at least one ground-truth box, so matching and IoU are
    meaningful;
  * 2 images with no ground-truth box, so the negative-set logic has something
    to score.
Every selected image also appears in the committed detection CSVs, which is what
lets the notebooks join a sample image to its real published detection.

Re-run (needs the original Roboflow exports, which are not public):

    python scripts/build_sample_dataset.py --source /path/to/Thesis_repo_work
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data" / "sample"

# Roboflow export folder name per amenity key
SOURCE_DIRS = {
    "bathtub": "Bathtub",
    "hairdryer": "Hairdryer",
    "kettle": "Kettle",
    "mirror": "Mirror",
    "tv": "Tv",
}

N_POSITIVE = 8
N_NEGATIVE = 2

# A small number of images in the TV split are public stock or AI-generated
# photographs rather than trivago listings: product shots, outdoor billboards and
# similar. They are fine to evaluate against, but redistributing them here would
# mean shipping third-party images whose licence we cannot establish, and would
# make the attribution file wrong. They are excluded from the committed sample,
# so everything under data/sample/ is trivago's and covered by their permission.
STOCK_NAME_PATTERN = re.compile(
    r"(shutterstock|istock|freepik|getty|depositphotos|alamy"
    r"|billboard|-is-shown-|generative-ai|LED_TV|_\d{3,}-\d{3,})",
    re.IGNORECASE,
)

# Windows refuses paths over 260 characters by default. A few source images carry
# very long descriptive names (one is 237 characters), and committing those would
# make `git clone` fail for anyone whose checkout directory is more than a couple
# of folders deep. Capping the file name leaves ~140 characters of headroom for
# the clone path plus `data/sample/<amenity>/`.
MAX_FILENAME = 120


def select(coco: dict, detected: set[str]) -> list[dict]:
    """Pick a deterministic slice: positives first, then negatives."""
    annotated = {a["image_id"] for a in coco["annotations"]}

    # Sorting by file name keeps the selection stable across runs and machines.
    images = sorted(coco["images"], key=lambda i: i["file_name"])
    usable = [
        i for i in images
        if i["file_name"] in detected
        and len(i["file_name"]) <= MAX_FILENAME
        and not STOCK_NAME_PATTERN.search(i["file_name"])
    ]

    positives = [i for i in usable if i["id"] in annotated][:N_POSITIVE]
    negatives = [i for i in usable if i["id"] not in annotated][:N_NEGATIVE]

    if len(positives) < N_POSITIVE:
        print(f"  warning: only {len(positives)} positives available", file=sys.stderr)
    return positives + negatives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Folder holding the Roboflow exports (Bathtub/, Kettle/, ...)")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"source not found: {args.source}", file=sys.stderr)
        return 1

    rows = []
    for key, folder in SOURCE_DIRS.items():
        split = args.source / folder / "test"
        ann_path = split / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"skip {key}: no annotations at {ann_path}", file=sys.stderr)
            continue

        det_csv = REPO_ROOT / "results" / "detection_outputs" / key / f"dets_{key}_baseline.csv"
        detected = set(pd.read_csv(det_csv)["image"].unique())

        coco = json.load(ann_path.open())
        chosen = select(coco, detected)
        chosen_ids = {i["id"] for i in chosen}

        out_dir = OUT_ROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)
        for image in chosen:
            shutil.copy2(split / image["file_name"], out_dir / image["file_name"])

        # The Roboflow export stamps a generic licence block onto every project.
        # It does not describe these images, which are trivago's, so it is
        # replaced rather than carried through.
        subset = {
            "info": {
                "description": f"{len(chosen)}-image sample of the {key} test split",
                "images_provided_by": "trivago N.V., reproduced with permission",
                "annotations": "hand-drawn for evaluation; no model was trained on them",
                "note": "Images are not covered by this repository's MIT licence. "
                        "See data/sample/ATTRIBUTION.md.",
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Provided by trivago N.V., all rights reserved",
                    "url": "",
                }
            ],
            "categories": coco["categories"],
            "images": chosen,
            "annotations": [a for a in coco["annotations"] if a["image_id"] in chosen_ids],
        }
        (out_dir / "_annotations.coco.json").write_text(json.dumps(subset, indent=1) + "\n")

        n_ann = len(subset["annotations"])
        print(f"{key:<10} {len(chosen)} images, {n_ann} boxes -> {out_dir.relative_to(REPO_ROOT)}")
        rows.append((key, len(chosen), n_ann))

    total_img = sum(r[1] for r in rows)
    attribution = [
        "# Image attribution",
        "",
        "## Images provided by trivago N.V.",
        "",
        f"The {total_img} photographs in this folder are real accommodation listing",
        "images from trivago's image inventory, reproduced with permission for this",
        "research. They remain the property of trivago N.V.",
        "",
        "They are **not** covered by this repository's MIT licence, which applies to",
        "the code only. If you fork this repository, the code is yours to reuse; the",
        "images are not.",
        "",
        "Bounding boxes were annotated by hand in a private Roboflow workspace and",
        "exported in COCO format. No annotated data was used to train or fine-tune any",
        "model. OWL-ViT runs zero-shot, and these annotations exist purely to",
        "evaluate it.",
        "",
        "| Amenity | Images | Ground-truth boxes |",
        "|:---|---:|---:|",
    ]
    for key, n_img, n_ann in rows:
        attribution.append(f"| {key} | {n_img} | {n_ann} |")
    attribution += [
        "",
        "Boxes can outnumber images: a single photograph often contains several",
        "instances of an amenity, which is why the two columns do not match.",
        "",
        "Images carrying no box are the **negative set**: the same kind of room",
        "without the target amenity, used to measure false positives.",
        "",
        "This is a deterministic slice of the test splits, committed so the notebooks",
        "run on a clean checkout. Full splits are not committed. See",
        "[`docs/dataset_inventory.md`](../../docs/dataset_inventory.md) for provenance,",
        "counts, and known data-quality issues.",
        "",
        "Regenerate with `python scripts/build_sample_dataset.py --source <exports>`.",
    ]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "ATTRIBUTION.md").write_text("\n".join(attribution) + "\n")
    print(f"\nwrote {(OUT_ROOT / 'ATTRIBUTION.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
