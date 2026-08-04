"""Tests for detection matching and question-quality metrics."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.detection.metrics import best_per_prompt, coco_boxes, evaluate_negatives
from src.evaluation import question_metrics as qm


class TestCocoBoxes:
    def test_converts_xywh_to_xyxy(self):
        boxes = coco_boxes([{"bbox": [10, 20, 30, 40]}])
        assert torch.allclose(boxes, torch.tensor([[10.0, 20.0, 40.0, 60.0]]))

    def test_empty_returns_zero_rows(self):
        assert coco_boxes([]).shape == (0, 4)


class TestNegatives:
    def test_flags_images_above_threshold(self):
        detections = pd.DataFrame(
            {"image": ["a.jpg", "b.jpg"], "score": [0.9, 0.01],
             "x1": [0, 0], "y1": [0, 0], "x2": [1, 1], "y2": [1, 1]}
        )
        result = evaluate_negatives(detections, ["a.jpg", "b.jpg"], [0.5])
        assert result.loc[0, "false_images"] == 1
        assert result.loc[0, "fpr"] == pytest.approx(0.5)
        assert result.loc[0, "specificity"] == pytest.approx(0.5)


class TestBestPerPrompt:
    def test_picks_highest_f1_per_prompt(self):
        summary = pd.DataFrame(
            {"prompt": ["a", "a", "b"], "conf": [0.05, 0.15, 0.05], "f1": [0.5, 0.8, 0.6]}
        )
        best = best_per_prompt(summary)
        assert len(best) == 2
        assert best.loc[best["prompt"] == "a", "f1"].item() == 0.8


class TestQuestionMetrics:
    def test_detects_fallbacks(self):
        assert qm.is_fallback("")
        assert qm.is_fallback("[FALLBACK] RateLimitError")
        assert qm.is_fallback(float("nan"))
        assert not qm.is_fallback("Is the kettle stainless steel?")

    def test_well_formed_requires_question_mark(self):
        assert qm.is_well_formed("Is the kettle made of stainless steel?")
        assert not qm.is_well_formed("The kettle is stainless steel.")
        assert not qm.is_well_formed("Kettle?")

    def test_distinct_n_penalises_repetition(self):
        varied = ["Is the tub deep?", "Does the room have a shower screen?"]
        repeated = ["Is the tub deep?"] * 5
        assert qm.distinct_n(varied, 2) > qm.distinct_n(repeated, 2)

    def test_duplicate_rate(self):
        assert qm.duplicate_rate(["a?", "a?", "b?"]) == pytest.approx(1 / 3)

    def test_summarize_has_expected_keys(self):
        result = qm.summarize(["Is the kettle clean?", ""])
        assert result["n"] == 2
        assert result["fallback_rate"] == pytest.approx(0.5)


class TestOverwriteGuard:
    """`--out-root` must not be able to reach results/ by spelling it differently.

    The guard originally compared Paths directly, so `--out-root results` was
    caught but an absolute path to the same directory was not, and a sample-scale
    run silently overwrote the published summary CSVs.
    """

    def test_relative_and_absolute_are_the_same_directory(self):
        from src.detection.run_detection import DEFAULT_OUT_ROOT, is_same_directory

        assert is_same_directory(DEFAULT_OUT_ROOT, DEFAULT_OUT_ROOT.resolve())
        assert is_same_directory(Path("results"), Path.cwd() / "results")
        assert is_same_directory(Path("results"), Path("./results"))
        assert is_same_directory(Path("results"), Path("results/../results"))

    def test_different_directories_are_not_confused(self):
        from src.detection.run_detection import is_same_directory

        assert not is_same_directory(Path("results"), Path("other-results"))
        assert not is_same_directory(Path("results"), Path("/tmp/elsewhere/results"))


class TestConfig:
    def test_all_amenities_load(self):
        configs = load_config()
        assert set(configs) == {"bathtub", "hairdryer", "kettle", "mirror", "tv"}

    def test_every_amenity_has_a_baseline_prompt(self):
        for cfg in load_config().values():
            assert "baseline" in cfg.prompt_sets
            assert cfg.prompts("baseline")

    def test_unknown_prompt_set_raises(self):
        cfg = load_config()["kettle"]
        with pytest.raises(KeyError):
            cfg.prompts("does-not-exist")
