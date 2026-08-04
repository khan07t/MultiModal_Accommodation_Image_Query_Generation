"""Tests for prompt construction."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.question_generation.prompts import (
    build_prompt,
    confidence_bucket,
    detection_summary,
)

ROW = {"image": "room_01.jpg", "label": "kettle", "score": 0.72, "source_prompt": "baseline"}


def test_confidence_buckets():
    assert confidence_bucket(0.9).startswith("high")
    assert confidence_bucket(0.45).startswith("medium")
    assert confidence_bucket(0.1).startswith("low")


def test_detection_summary_includes_label_and_image():
    summary = detection_summary(ROW)
    assert "kettle" in summary
    assert "room_01.jpg" in summary


def test_build_prompt_appends_profile_context():
    generic = build_prompt(detection_summary(ROW), profile="generic")
    family = build_prompt(detection_summary(ROW), profile="family")
    assert "children" in family
    assert "children" not in generic
    assert family.endswith("Question:")


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        build_prompt("summary", profile="astronaut")
