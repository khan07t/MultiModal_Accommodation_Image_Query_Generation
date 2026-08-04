"""Detection evaluation: greedy IoU matching, threshold sweeps, negative sets.

This consolidates the `evaluate_coco` / `negatives_eval` blocks that previously
appeared once per amenity notebook.

torch and torchvision are imported lazily, inside the three functions that
actually need them. Everything else here is pure pandas, which keeps the
"reproduce the published table without a GPU" path in the README honest.
`scripts/build_results_table.py` and `run_detection --help` both work with no
deep-learning stack installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

EPS = 1e-9


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    mean_iou: float
    tp: int
    fp: int
    fn: int
    conf: float
    prompt: str

    def as_dict(self) -> dict:
        return asdict(self)


def coco_boxes(annotations: Sequence[dict]) -> "torch.Tensor":
    """Convert COCO [x, y, w, h] annotations to an (N, 4) xyxy tensor."""
    import torch

    if not annotations:
        return torch.zeros((0, 4), dtype=torch.float32)
    boxes = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
    return torch.tensor(boxes, dtype=torch.float32)


def _predicted_boxes(rows: pd.DataFrame, conf_threshold: float) -> "torch.Tensor":
    import torch

    kept = rows.loc[rows["score"] > conf_threshold, ["x1", "y1", "x2", "y2"]]
    if kept.empty:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(kept.to_numpy(dtype=np.float32), dtype=torch.float32)


def evaluate_detections(
    detections: pd.DataFrame,
    coco,
    conf_threshold: float,
    iou_threshold: float = 0.50,
    prompt: str = "baseline",
) -> DetectionMetrics:
    """Score a detection CSV against COCO ground truth at one confidence level.

    Predictions are matched greedily: each prediction takes its best-IoU ground
    truth box. A prediction above `iou_threshold` counts as a true positive,
    otherwise a false positive. Unmatched ground truth boxes count as false
    negatives. This mirrors the thesis evaluation exactly.
    """
    import torch
    from torchvision.ops import box_iou

    tp = fp = fn = 0
    ious: List[float] = []

    grouped = {name: frame for name, frame in detections.groupby("image")}

    for image_id in coco.getImgIds():
        info = coco.loadImgs(image_id)[0]
        gt = coco_boxes(coco.loadAnns(coco.getAnnIds(imgIds=image_id)))

        rows = grouped.get(info["file_name"])
        pred = (
            _predicted_boxes(rows, conf_threshold)
            if rows is not None
            else torch.zeros((0, 4), dtype=torch.float32)
        )

        if len(pred) == 0:
            fn += len(gt)
            continue
        if len(gt) == 0:
            fp += len(pred)
            continue

        max_iou, _ = box_iou(pred, gt).max(dim=1)
        ious.extend(max_iou.tolist())

        matched = int((max_iou > iou_threshold).sum().item())
        tp += matched
        fp += len(pred) - matched
        fn += max(0, len(gt) - matched)

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        tp=tp,
        fp=fp,
        fn=fn,
        conf=conf_threshold,
        prompt=prompt,
    )


def sweep_thresholds(
    detections_by_prompt: Dict[str, pd.DataFrame],
    coco,
    conf_sweep: Iterable[float],
    iou_threshold: float = 0.50,
) -> pd.DataFrame:
    """Evaluate every prompt set across every confidence threshold."""
    rows = []
    for prompt, detections in detections_by_prompt.items():
        for conf in conf_sweep:
            metrics = evaluate_detections(
                detections, coco, conf, iou_threshold=iou_threshold, prompt=prompt
            )
            rows.append(metrics.as_dict())
    return pd.DataFrame(rows)


def evaluate_negatives(
    detections: pd.DataFrame,
    negative_images: Sequence[str],
    conf_sweep: Iterable[float],
) -> pd.DataFrame:
    """False-positive rate and specificity on images known to contain no target.

    An image counts as a false positive if any detection survives the threshold.
    """
    negatives = set(negative_images)
    total = max(1, len(negatives))
    subset = detections[detections["image"].isin(negatives)]

    rows = []
    for conf in conf_sweep:
        kept = subset[subset["score"] > conf]
        false_images = kept["image"].nunique()
        fpr = false_images / total
        rows.append(
            {
                "conf": conf,
                "fpr": fpr,
                "specificity": 1.0 - fpr,
                "false_images": false_images,
                "mean_conf": float(kept["score"].mean()) if not kept.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def best_per_prompt(summary: pd.DataFrame) -> pd.DataFrame:
    """Highest-F1 row for each prompt set."""
    return (
        summary.sort_values(["prompt", "f1"], ascending=[True, False])
        .groupby("prompt", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def load_summary(path: Path | str) -> pd.DataFrame:
    """Read a committed summary CSV (utf-8-sig handles the Excel BOM)."""
    return pd.read_csv(path, encoding="utf-8-sig")
