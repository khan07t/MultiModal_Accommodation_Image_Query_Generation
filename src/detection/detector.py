"""OWL-ViT zero-shot detection wrapper.

Replaces the per-amenity `infer_image` / `run_prompt_set` blocks that were
duplicated across five notebooks. The model is loaded once and reused for every
amenity and prompt set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

DETECTION_COLUMNS = ["image", "label", "score", "x1", "y1", "x2", "y2"]


def resolve_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Detection:
    image: str
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


class OwlViTDetector:
    """Thin wrapper around OWL-ViT for text-prompted object detection."""

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch16",
        device: str | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model_name = model_name
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        prompts: Sequence[str],
        threshold: float = 0.05,
    ) -> List[Detection]:
        """Run detection on one image for one prompt set."""
        inputs = self.processor(text=list(prompts), images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        processed = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        results: List[Detection] = []
        for box, score, label in zip(
            processed["boxes"].cpu(),
            processed["scores"].cpu(),
            processed["labels"].cpu(),
        ):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            results.append(
                Detection(
                    image="",
                    label=prompts[int(label)],
                    score=float(score),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return results

    def detect_folder(
        self,
        image_paths: Iterable[Path],
        prompts: Sequence[str],
        threshold: float = 0.05,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Run detection over many images and return a tidy DataFrame."""
        paths = list(image_paths)
        iterator: Iterable[Path] = paths
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(paths, desc=f"OWL-ViT [{prompts[0]}]")
            except ImportError:  # pragma: no cover - tqdm is optional
                pass

        rows: List[dict] = []
        for path in iterator:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            for det in self.detect(image, prompts, threshold=threshold):
                det.image = path.name
                rows.append(vars(det))

        return pd.DataFrame(rows, columns=DETECTION_COLUMNS)
