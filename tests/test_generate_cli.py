"""Tests for the question-generation CLI.

The vision path used to be advertised in `--help` while always raising, because
`generate.py` called `backend.generate()` for every backend and
`GeminiVisionBackend.generate()` refuses to run without an image. Nothing caught
it: the class had unit tests, but no test exercised the CLI wiring. These tests
use a fake backend so the dispatch is checked without an API key or a network.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.question_generation import generate as cli
from src.question_generation.backends import FALLBACK_MARKER


class FakeVisionBackend:
    """Stands in for GeminiVisionBackend. Records what it was asked."""

    name = "fake-vision"

    def __init__(self):
        self.calls = []

    def generate(self, prompts):
        raise NotImplementedError("vision backend needs an image")

    def generate_for_detection(self, image_path, box, label, num_questions=3,
                               perspective="single"):
        self.calls.append(
            {"image_path": Path(image_path), "box": list(box), "label": label,
             "num_questions": num_questions, "perspective": perspective}
        )
        return [f"{label} question {i} for {perspective}?" for i in range(num_questions)]


def detections_frame():
    return pd.DataFrame([
        {"image": "a.jpg", "label": "bathtub", "score": 0.9,
         "x1": 10, "y1": 20, "x2": 110, "y2": 120},
        {"image": "b.jpg", "label": "bathtub", "score": 0.8,
         "x1": 0, "y1": 0, "x2": 50, "y2": 50},
    ])


class TestVisionDispatch:
    def test_calls_generate_for_detection_not_generate(self, tmp_path):
        """The regression guard: the vision path must not touch generate()."""
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        backend = FakeVisionBackend()

        answers = cli.generate_vision(
            detections_frame(), backend, tmp_path, "couple", 3
        )

        assert len(answers) == 2
        assert len(backend.calls) == 2
        assert all(FALLBACK_MARKER not in a for a in answers)

    def test_passes_box_label_and_perspective_through(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        backend = FakeVisionBackend()

        cli.generate_vision(detections_frame(), backend, tmp_path, "group", 2)

        first = backend.calls[0]
        assert first["box"] == [10, 20, 110, 120]
        assert first["label"] == "bathtub"
        assert first["perspective"] == "group"
        assert first["num_questions"] == 2

    def test_missing_image_becomes_a_fallback_not_a_crash(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")  # b.jpg deliberately absent
        backend = FakeVisionBackend()

        answers = cli.generate_vision(detections_frame(), backend, tmp_path, "single", 3)

        assert len(answers) == 2
        assert FALLBACK_MARKER in answers[1]
        assert len(backend.calls) == 1


class TestCliArguments:
    def test_vision_backend_without_image_root_exits_nonzero(self, tmp_path):
        detections = tmp_path / "dets.csv"
        detections_frame().to_csv(detections, index=False)

        code = cli.main([
            "--detections", str(detections),
            "--backend", "gemini-vision",
            "--out", str(tmp_path / "out.csv"),
        ])
        assert code == 1

    def test_text_backend_dry_run_still_works(self, tmp_path):
        detections = tmp_path / "dets.csv"
        detections_frame().to_csv(detections, index=False)
        out = tmp_path / "out.csv"

        code = cli.main([
            "--detections", str(detections),
            "--backend", "flan-t5-base",
            "--out", str(out),
            "--dry-run",
        ])
        assert code == 0
        written = pd.read_csv(out)
        assert len(written) == 2
        assert "llm_prompt" in written.columns

    def test_parser_advertises_the_vision_flags(self):
        options = {a.dest for a in cli.build_parser()._actions}
        assert {"image_root", "perspective", "num_questions"} <= options
