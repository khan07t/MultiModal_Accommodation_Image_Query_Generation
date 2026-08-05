"""Tests for detection matching and question-quality metrics.

No torch import: the evaluation path is pure NumPy, so this suite runs on a
machine with no deep-learning stack installed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.detection.metrics import (
    best_per_prompt,
    box_iou,
    coco_boxes,
    evaluate_detections,
    evaluate_negatives,
    match_one_to_one,
    nms,
)
from src.evaluation import question_metrics as qm


class FakeCoco:
    """Minimal stand-in for pycocotools.COCO, so scoring is testable offline."""

    def __init__(self, annotations_by_image):
        self._names = list(annotations_by_image)
        self._boxes = annotations_by_image

    def getImgIds(self):
        return list(range(len(self._names)))

    def loadImgs(self, image_id):
        return [{"file_name": self._names[image_id]}]

    def getAnnIds(self, imgIds):
        return [imgIds]

    def loadAnns(self, ids):
        name = self._names[ids[0]]
        return [{"bbox": [x1, y1, x2 - x1, y2 - y1]}
                for x1, y1, x2, y2 in self._boxes[name]]


def detections_frame(rows):
    return pd.DataFrame(rows, columns=["image", "score", "x1", "y1", "x2", "y2"])


class TestCocoBoxes:
    def test_converts_xywh_to_xyxy(self):
        boxes = coco_boxes([{"bbox": [10, 20, 30, 40]}])
        assert np.allclose(boxes, np.array([[10.0, 20.0, 40.0, 60.0]]))

    def test_empty_returns_zero_rows(self):
        assert coco_boxes([]).shape == (0, 4)


class TestBoxIou:
    def test_identical_boxes_score_one(self):
        box = np.array([[0.0, 0.0, 10.0, 10.0]])
        assert box_iou(box, box)[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[20.0, 20.0, 30.0, 30.0]])
        assert box_iou(a, b)[0, 0] == pytest.approx(0.0)

    def test_half_overlap(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[5.0, 0.0, 15.0, 10.0]])
        assert box_iou(a, b)[0, 0] == pytest.approx(1 / 3, abs=1e-6)


class TestNms:
    def test_suppresses_near_duplicates(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
        assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5).tolist()) == [0, 2]

    def test_keeps_the_higher_scoring_duplicate(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
        assert nms(boxes, np.array([0.3, 0.9]), 0.5).tolist() == [1]

    def test_empty_input(self):
        assert len(nms(np.zeros((0, 4)), np.zeros(0))) == 0


class TestOneToOneMatching:
    """Regression guard for the many-to-one bug corrected in February 2026."""

    def test_duplicate_predictions_claim_one_box_each(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0]])
        preds = np.array([[0, 0, 10, 10]] * 3, dtype=float)
        matched, ious = match_one_to_one(preds, np.array([0.9, 0.8, 0.7]), gt)
        assert matched == 1
        assert len(ious) == 1

    def test_two_objects_two_predictions(self):
        gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
        preds = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
        assert match_one_to_one(preds, np.array([0.9, 0.8]), gt)[0] == 2

    def test_below_iou_threshold_does_not_match(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0]])
        preds = np.array([[9.0, 9.0, 19.0, 19.0]])
        assert match_one_to_one(preds, np.array([0.9]), gt)[0] == 0


class TestEvaluateDetections:
    def test_tp_plus_fn_equals_ground_truth(self):
        """The invariant that exposed the original bug.

        Ground truth is a fixed property of the dataset, so tp + fn must equal
        the box count at every confidence threshold. Under many-to-one matching
        it did not: mirror reported 420 against 285 annotated boxes.
        """
        coco = FakeCoco({
            "a.jpg": [[0, 0, 10, 10], [50, 50, 60, 60]],
            "b.jpg": [[0, 0, 10, 10]],
        })
        detections = detections_frame([
            ("a.jpg", 0.9, 0, 0, 10, 10),
            ("a.jpg", 0.8, 1, 1, 11, 11),
            ("a.jpg", 0.7, 0, 0, 10, 10),
            ("a.jpg", 0.6, 50, 50, 60, 60),
            ("b.jpg", 0.5, 0, 0, 10, 10),
        ])
        for conf in (0.05, 0.15, 0.25, 0.45):
            m = evaluate_detections(detections, coco, conf)
            assert m.tp + m.fn == 3, f"tp+fn != 3 at conf={conf}"

    def test_duplicates_become_false_positives_without_nms(self):
        coco = FakeCoco({"a.jpg": [[0, 0, 10, 10]]})
        detections = detections_frame([
            ("a.jpg", 0.9, 0, 0, 10, 10),
            ("a.jpg", 0.8, 0, 0, 10, 10),
        ])
        m = evaluate_detections(detections, coco, 0.05, apply_nms=False)
        assert (m.tp, m.fp, m.fn) == (1, 1, 0)

    def test_nms_removes_the_duplicate_entirely(self):
        coco = FakeCoco({"a.jpg": [[0, 0, 10, 10]]})
        detections = detections_frame([
            ("a.jpg", 0.9, 0, 0, 10, 10),
            ("a.jpg", 0.8, 0, 0, 10, 10),
        ])
        m = evaluate_detections(detections, coco, 0.05, apply_nms=True)
        assert (m.tp, m.fp, m.fn) == (1, 0, 0)

    def test_missed_object_counts_as_false_negative(self):
        coco = FakeCoco({"a.jpg": [[0, 0, 10, 10]]})
        m = evaluate_detections(detections_frame([]), coco, 0.05)
        assert (m.tp, m.fp, m.fn) == (0, 0, 1)

    def test_mean_iou_uses_matched_pairs_only(self):
        coco = FakeCoco({"a.jpg": [[0.0, 0.0, 10.0, 10.0]]})
        detections = detections_frame([
            ("a.jpg", 0.9, 0, 0, 10, 10),
            ("a.jpg", 0.8, 500, 500, 510, 510),
        ])
        assert evaluate_detections(detections, coco, 0.05).mean_iou == pytest.approx(1.0, abs=1e-6)


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
