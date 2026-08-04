"""Configuration loading for the amenity detection and query generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "amenities.yaml"


@dataclass
class AmenityConfig:
    """Everything the pipeline needs to know about one amenity class."""

    key: str
    display_name: str
    dataset_slug: str
    prompt_sets: Dict[str, List[str]]
    model_name: str
    inference_threshold: float
    conf_sweep: List[float]
    neg_conf_sweep: List[float]
    eval_iou_threshold: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def prompts(self, prompt_set: str) -> List[str]:
        if prompt_set not in self.prompt_sets:
            available = ", ".join(sorted(self.prompt_sets))
            raise KeyError(
                f"Prompt set {prompt_set!r} not defined for {self.key!r}. "
                f"Available: {available}"
            )
        return list(self.prompt_sets[prompt_set])


def load_config(path: Path | str = DEFAULT_CONFIG) -> Dict[str, AmenityConfig]:
    """Load `configs/amenities.yaml` into AmenityConfig objects keyed by amenity."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    defaults = raw.get("defaults", {})
    configs: Dict[str, AmenityConfig] = {}

    for key, entry in raw["amenities"].items():
        merged = {**defaults, **entry}
        configs[key] = AmenityConfig(
            key=key,
            display_name=merged.get("display_name", key.title()),
            dataset_slug=merged.get("dataset_slug", key),
            prompt_sets=merged["prompt_sets"],
            model_name=merged["model_name"],
            inference_threshold=float(merged["inference_threshold"]),
            conf_sweep=[float(c) for c in merged["conf_sweep"]],
            neg_conf_sweep=[float(c) for c in merged["neg_conf_sweep"]],
            eval_iou_threshold=float(merged["eval_iou_threshold"]),
        )

    return configs


def get_amenity(key: str, path: Path | str = DEFAULT_CONFIG) -> AmenityConfig:
    """Load a single amenity configuration by key."""
    configs = load_config(path)
    if key not in configs:
        available = ", ".join(sorted(configs))
        raise KeyError(f"Unknown amenity {key!r}. Available: {available}")
    return configs[key]
