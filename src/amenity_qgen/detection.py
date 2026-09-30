"""Zero-shot amenity detection with OWL-ViT and COCO-style evaluation."""

import json
import os

import numpy as np
import pandas as pd

from .config import IOU_THRESHOLD, OWLVIT_MODEL


# ---------------------------------------------------------------- inference

def load_owlvit(model_name: str = OWLVIT_MODEL, device: str = None):
    import torch
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()
    return processor, model, device


def infer_image(image, categories, processor, model, device, score_threshold=0.05):
    """Run OWL-ViT on one PIL image for a list of text prompts."""
    import torch

    inputs = processor(text=categories, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    dets = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=score_threshold
    )[0]
    return {k: v.detach().cpu() for k, v in dets.items()}


def run_prompt_set(image_dir, coco_json, categories, processor, model, device,
                   score_threshold=0.10):
    """Detect on every test image and return one row per predicted box."""
    from PIL import Image

    coco = json.load(open(coco_json))
    rows = []
    for info in coco["images"]:
        image = Image.open(os.path.join(image_dir, info["file_name"])).convert("RGB")
        dets = infer_image(image, categories, processor, model, device, score_threshold)
        for box, score, label in zip(dets["boxes"], dets["scores"], dets["labels"]):
            x1, y1, x2, y2 = box.tolist()
            rows.append(dict(image=info["file_name"], label=categories[int(label)],
                             score=float(score), x1=x1, y1=y1, x2=x2, y2=y2))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- evaluation

def load_ground_truth(coco_json):
    """Boxes per image (xyxy). Images labelled 'none' are kept as negatives."""
    coco = json.load(open(coco_json))
    keep = {c["id"] for c in coco["categories"] if c["name"].lower() != "none"}
    names = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = {name: [] for name in names.values()}
    for a in coco["annotations"]:
        if a["category_id"] in keep:
            x, y, w, h = a["bbox"]
            gt[names[a["image_id"]]].append([x, y, x + w, y + h])
    return {k: np.array(v, dtype=float).reshape(-1, 4) for k, v in gt.items()}


def box_iou(a, b):
    a, b = a[:, None, :], b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-12)


def evaluate_detections(dets, gt, conf_threshold=0.05, iou_thr=IOU_THRESHOLD):
    """Precision, recall, F1 and mean IoU at one confidence threshold.

    Boxes are matched to annotations in descending score order and each
    annotation can be matched once. Unmatched boxes are false positives,
    unmatched annotations false negatives. Images without annotations are
    included, so any box there counts against precision.
    """
    tp = fp = fn = 0
    ious = []
    for image, g in gt.items():
        rows = dets[(dets["image"] == image) & (dets["score"] > conf_threshold)]
        rows = rows.sort_values("score", ascending=False)
        p = rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)

        if len(p) == 0:
            fn += len(g)
            continue
        if len(g) == 0:
            fp += len(p)
            continue

        m = box_iou(p, g)
        ious += m.max(axis=1).tolist()
        used = set()
        for i in range(len(p)):
            best, best_iou = -1, iou_thr
            for j in range(len(g)):
                if j not in used and m[i, j] > best_iou:
                    best, best_iou = j, m[i, j]
            if best >= 0:
                used.add(best)
        tp += len(used)
        fp += len(p) - len(used)
        fn += len(g) - len(used)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1,
                mean_iou=float(np.mean(ious)) if ious else 0.0, tp=tp, fp=fp, fn=fn)


def threshold_sweep(dets, gt, thresholds, prompt=None):
    out = [dict(conf=c, **evaluate_detections(dets, gt, c)) for c in thresholds]
    df = pd.DataFrame(out)
    if prompt:
        df.insert(0, "prompt", prompt)
    return df


def best_operating_point(sweep):
    """Row with the highest F1 for each prompt."""
    return sweep.loc[sweep.groupby("prompt")["f1"].idxmax()].sort_values("f1", ascending=False)


def negative_image_rate(dets, negative_images, thresholds):
    """Share of amenity-free images that still get at least one box."""
    rows = []
    for c in thresholds:
        flagged = dets[(dets["score"] > c) & dets["image"].isin(negative_images)]["image"].nunique()
        n = len(negative_images)
        rows.append(dict(conf=c, false_images=flagged, fpr=flagged / n if n else 0.0,
                         specificity=1 - flagged / n if n else 1.0))
    return pd.DataFrame(rows)
