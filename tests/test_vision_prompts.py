"""Tests for vision-conditioned prompt construction, cropping, and parsing.

These cover the parts of the vision path that run without an API key or a GPU:
prompt assembly, box padding, and parsing a numbered model response.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.question_generation.prompts import (
    PERSPECTIVES,
    build_vision_prompt,
    pad_box,
    parse_numbered_questions,
)


class TestBuildVisionPrompt:
    def test_includes_label_and_count(self):
        prompt = build_vision_prompt("bathtub", num_questions=3)
        assert "bathtub" in prompt
        assert "numbered 1-3" in prompt

    def test_perspective_context_differs(self):
        single = build_vision_prompt("kettle", perspective="single")
        group = build_vision_prompt("kettle", perspective="group")
        assert single != group
        assert PERSPECTIVES["group"] in group

    def test_unknown_perspective_raises(self):
        with pytest.raises(KeyError):
            build_vision_prompt("kettle", perspective="astronaut")

    def test_zero_questions_raises(self):
        with pytest.raises(ValueError):
            build_vision_prompt("kettle", num_questions=0)


class TestPadBox:
    def test_expands_by_fraction(self):
        # 100x100 box at (100, 100), 35% padding -> 35px each side
        assert pad_box([100, 100, 200, 200], (1000, 1000), 0.35) == (65, 65, 235, 235)

    def test_clips_to_image_bounds(self):
        assert pad_box([10, 10, 110, 110], (120, 120), 0.35) == (0, 0, 120, 120)

    def test_zero_padding_is_identity(self):
        assert pad_box([10, 20, 30, 40], (100, 100), 0.0) == (10, 20, 30, 40)


class TestParseNumberedQuestions:
    def test_strips_numbering_and_quotes(self):
        raw = '1. "Is the tub full-size?"\n2) Does it have jets?\n3: Is there a shower?'
        parsed = parse_numbered_questions(raw, 3, "bathtub")
        assert parsed == [
            "Is the tub full-size?",
            "Does it have jets?",
            "Is there a shower?",
        ]

    def test_appends_missing_question_mark(self):
        parsed = parse_numbered_questions("1. Is the kettle descaled", 1, "kettle")
        assert parsed[0].endswith("?")

    def test_pads_short_responses_with_fallbacks(self):
        parsed = parse_numbered_questions("1. Is the tub full-size?", 3, "bathtub")
        assert len(parsed) == 3
        assert all("bathtub" in q for q in parsed[1:])

    def test_truncates_long_responses(self):
        raw = "\n".join(f"{i}. Is this question number {i} about the tub?" for i in range(1, 8))
        assert len(parse_numbered_questions(raw, 3, "bathtub")) == 3

    def test_drops_parse_noise(self):
        # "Questions:" and stray short lines are remnants, not questions
        raw = "Questions:\n\n1. Is the bathtub slip-resistant?\n-\n2. Is it a walk-in?"
        parsed = parse_numbered_questions(raw, 2, "bathtub")
        assert parsed == ["Is the bathtub slip-resistant?", "Is it a walk-in?"]

    def test_empty_response_is_all_fallback(self):
        parsed = parse_numbered_questions("", 3, "mirror")
        assert len(parsed) == 3
        assert all(q.endswith("?") for q in parsed)
