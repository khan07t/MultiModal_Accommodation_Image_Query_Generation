"""Detection evaluation: one-to-one IoU matching, threshold sweeps, negative sets.

This consolidates the `evaluate_coco` / `negatives_eval` blocks that previously
appeared once per amenity notebook.

Everything here is pure NumPy and pandas. No torch, no torchvision, so
`scripts/build_results_table.py`, `pytest tests/` and `run_detection --help` all
work with no deep-learning stack installed. Only the detector itself needs torch.

Correction, February 2026
-------------------------
An earlier version of this module scored detections like this:

    max_iou, _ = box_iou(pred, gt).max(dim=1)
    matched = (max_iou > iou_threshold).sum()

That counts *predictions that overlap something*, not *ground-truth boxes that
were found*. Nothing stopped several predictions claiming the same box, so three
boxes drawn on one bathtub scored three true positives and zero false positives.
`max(0, len(gt) - matched)` then clamped false negatives to zero whenever
duplicates outnumbered the ground truth, hiding real misses.

The invariant that exposes it: **`tp + fn` must equal the ground-truth box count**
at every threshold and for every prompt set, because ground truth is a fixed
property of the dataset. It did not. Mirror reported 420 against 285 annotated
boxes. `tests/test_metrics.py::test_tp_plus_fn_equals_ground_truth` now asserts
it, so this cannot regress silently.

The bias scaled with predictions per image, so multi-synonym prompt sets were
inflated most. Correcting it therefore widened the gap in `baseline`'s favour
rather than narrowing it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9
DEFAULT_NMS_IOU = 0.50


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


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes, shape (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return intersection / (area_a[:, None] + area_b[None, :] - intersection + EPS)


def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_threshold: float = DEFAULT_NMS_IOU) -> np.ndarray:
    """Class-agnostic non-maximum suppression. Returns indices to keep.

    OWL-ViT often returns several boxes on the same object, especially when a
    prompt set contains multiple synonyms. The original pipeline had no
    suppression step at all, so those duplicates reached the scorer.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=int)

    order = np.argsort(-scores)
    keep: List[int] = []
    while len(order) > 0:
        current = order[0]
        keep.append(int(current))
        if len(order) == 1:
            break
        ious = box_iou(boxes[current][None, :], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=int)


def coco_boxes(annotations: Sequence[dict]) -> np.ndarray:
    """Convert COCO [x, y, w, h] annotations to an (N, 4) xyxy array."""
    if not annotations:
        return np.zeros((0, 4), dtype=float)
    boxes = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
    return np.asarray(boxes, dtype=float)


def _predicted(rows: pd.DataFrame, conf_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Boxes and scores above the threshold, clipped to non-negative coordinates."""
    kept = rows.loc[rows["score"] > conf_threshold, ["x1", "y1", "x2", "y2", "score"]]
    if kept.empty:
        return np.zeros((0, 4), dtype=float), np.zeros(0, dtype=float)
    boxes = np.clip(kept[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float), 0, None)
    return boxes, kept["score"].to_numpy(dtype=float)


def match_one_to_one(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.50,
) -> Tuple[int, List[float]]:
    """Greedy one-to-one assignment in descending score order.

    The most confident prediction claims its best unclaimed ground-truth box, and
    each box can be claimed only once, so N predictions on one object yield one
    true positive and N-1 false positives.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0, []

    ious = box_iou(pred_boxes, gt_boxes)
    claimed: set[int] = set()
    matched_ious: List[float] = []

    for pred_index in np.argsort(-pred_scores):
        row = ious[pred_index]
        best_gt, best_iou = -1, iou_threshold
        for gt_index in range(len(gt_boxes)):
            if gt_index in claimed:
                continue
            if row[gt_index] > best_iou:
                best_gt, best_iou = gt_index, row[gt_index]
        if best_gt >= 0:
            claimed.add(best_gt)
            matched_ious.append(float(best_iou))

    return len(matched_ious), matched_ious


def evaluate_detections(
    detections: pd.DataFrame,
    coco,
    conf_threshold: float,
    iou_threshold: float = 0.50,
    prompt: str = "baseline",
    apply_nms: bool = True,
    nms_iou: float = DEFAULT_NMS_IOU,
) -> DetectionMetrics:
    """Score a detection CSV against COCO ground truth at one confidence level.

    Class-agnostic NMS suppresses duplicate boxes, then predictions are matched
    one-to-one against ground truth in descending score order. Unmatched
    predictions are false positives, unclaimed ground-truth boxes are false
    negatives, and `tp + fn` therefore always equals the ground-truth count.

    `mean_iou` averages over matched pairs only. Including false positives, which
    have no meaningful IoU, would drag the figure toward zero for reasons that
    have nothing to do with localisation quality.
    """
    tp = fp = fn = 0
    ious: List[float] = []

    grouped = {name: frame for name, frame in detections.groupby("image")}

    for image_id in coco.getImgIds():
        info = coco.loadImgs(image_id)[0]
        gt = coco_boxes(coco.loadAnns(coco.getAnnIds(imgIds=image_id)))

        rows = grouped.get(info["file_name"])
        if rows is None:
            fn += len(gt)
            continue

        boxes, scores = _predicted(rows, conf_threshold)
        if apply_nms and len(boxes):
            keep = nms(boxes, scores, nms_iou)
            boxes, scores = boxes[keep], scores[keep]

        if len(boxes) == 0:
            fn += len(gt)
            continue
        if len(gt) == 0:
            fp += len(boxes)
            continue

        matched, matched_ious = match_one_to_one(boxes, scores, gt, iou_threshold)
        ious.extend(matched_ious)

        tp += matched
        fp += len(boxes) - matched
        fn += len(gt) - matched

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
    apply_nms: bool = True,
) -> pd.DataFrame:
    """Evaluate every prompt set across every confidence threshold."""
    rows = []
    for prompt, detections in detections_by_prompt.items():
        for conf in conf_sweep:
            metrics = evaluate_detections(
                detections, coco, conf,
                iou_threshold=iou_threshold, prompt=prompt, apply_nms=apply_nms,
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
    This counts images rather than matched pairs, so it was unaffected by the
    matching correction described at the top of this module.
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
