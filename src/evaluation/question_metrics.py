"""Quality metrics for generated amenity questions.

Covers the three measures reported in the thesis: fallback rate (how often a
backend failed to produce a usable question), lexical distinctiveness (how much
the questions vary rather than repeating a template), and basic well-formedness.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence

FALLBACK_PATTERNS = (
    "[fallback]",
    "i cannot",
    "i'm unable",
    "as an ai",
    "n/a",
)

_WORD = re.compile(r"[a-z']+")


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def is_fallback(question: str | float) -> bool:
    """True when the backend did not return a usable question."""
    if not isinstance(question, str) or not question.strip():
        return True
    lowered = question.lower()
    return any(pattern in lowered for pattern in FALLBACK_PATTERNS)


def fallback_rate(questions: Sequence[str]) -> float:
    if not len(questions):
        return 0.0
    return sum(is_fallback(q) for q in questions) / len(questions)


def is_well_formed(question: str) -> bool:
    """A usable question ends with '?' and is neither trivially short nor rambling."""
    if is_fallback(question):
        return False
    stripped = question.strip()
    return stripped.endswith("?") and 4 <= len(tokenize(stripped)) <= 30


def well_formed_rate(questions: Sequence[str]) -> float:
    if not len(questions):
        return 0.0
    return sum(is_well_formed(q) for q in questions) / len(questions)


def distinct_n(questions: Iterable[str], n: int = 2) -> float:
    """Distinct-n: unique n-grams divided by total n-grams.

    Higher means more varied output; a value near zero means the model is
    emitting one template repeatedly.
    """
    total = 0
    unique = set()
    for question in questions:
        if is_fallback(question):
            continue
        tokens = tokenize(question)
        for i in range(max(0, len(tokens) - n + 1)):
            gram = tuple(tokens[i : i + n])
            unique.add(gram)
            total += 1
    return len(unique) / total if total else 0.0


def type_token_ratio(questions: Iterable[str]) -> float:
    """Vocabulary richness across the whole question set."""
    tokens: List[str] = []
    for question in questions:
        if not is_fallback(question):
            tokens.extend(tokenize(question))
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def duplicate_rate(questions: Sequence[str]) -> float:
    """Share of questions that are exact repeats of an earlier question."""
    usable = [q.strip().lower() for q in questions if not is_fallback(q)]
    if not usable:
        return 0.0
    counts = Counter(usable)
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    return repeats / len(usable)


def summarize(questions: Sequence[str]) -> dict:
    """All question-quality metrics for one backend/profile combination."""
    return {
        "n": len(questions),
        "fallback_rate": round(fallback_rate(questions), 4),
        "well_formed_rate": round(well_formed_rate(questions), 4),
        "distinct_1": round(distinct_n(questions, 1), 4),
        "distinct_2": round(distinct_n(questions, 2), 4),
        "type_token_ratio": round(type_token_ratio(questions), 4),
        "duplicate_rate": round(duplicate_rate(questions), 4),
    }
