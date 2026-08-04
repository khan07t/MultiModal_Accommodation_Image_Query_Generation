"""Prompt construction for amenity question generation.

Two conditioning modes:

*Text-only.* A detection row becomes a short natural-language summary, which is
wrapped in a system instruction and sent to a generation backend. This is the
mode that produces the template collapse documented as Finding 3 in the README:
the summary is nearly identical across rows, so the output is too.

*Vision.* The detected region is cropped from the source photograph and passed
to a multimodal model alongside the instruction. The model sees the amenity
rather than a description of it, which is what breaks the collapse.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

SYSTEM_INSTRUCTION = (
    "You help travellers evaluate accommodation listings. "
    "Given a detected amenity in a hotel photo, write ONE short question a guest "
    "might want answered before booking. Keep it factual, neutral, answerable "
    "from the photo or the listing, and grounded in the detected object. "
    "Return only the question."
)

# Traveller profiles used in the profile-conditioned experiments.
PROFILES: Mapping[str, str] = {
    "generic": "",
    "family": "The guest is travelling with young children and cares about safety and space.",
    "business": "The guest is travelling for work and cares about efficiency and connectivity.",
    "accessibility": "The guest has mobility needs and cares about accessible fixtures.",
    "luxury": "The guest is looking for premium finishes and comfort.",
}


def confidence_bucket(score: float) -> str:
    """Map a raw detection score onto a coarse verbal band.

    Passing the raw float to an LLM invited spurious precision in the generated
    text, so the thesis buckets it instead.
    """
    if score >= 0.60:
        return f"high confidence ({score:.2f})"
    if score >= 0.30:
        return f"medium confidence ({score:.2f})"
    return f"low confidence ({score:.2f})"


def detection_summary(row: Mapping) -> str:
    """Turn one detection row into the textual summary shown to the model."""
    return (
        f"Image file: {row['image']}. "
        f"Detected amenity: {row['label']}. "
        f"Detection confidence: {confidence_bucket(float(row['score']))}. "
        f"Source prompt type: {row.get('source_prompt', 'baseline')}."
    )


def build_prompt(summary: str, profile: str = "generic") -> str:
    """Assemble the final prompt string for a generation backend."""
    if profile not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown profile {profile!r}. Available: {available}")

    parts = [SYSTEM_INSTRUCTION]
    if PROFILES[profile]:
        parts.append(PROFILES[profile])
    parts.append(f"Detection summary: {summary}")
    parts.append("Question:")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Vision-conditioned generation
# --------------------------------------------------------------------------

# Travelling-party perspectives used in the vision runs. The committed
# `*_gemini3_perspectives.csv` files carry three questions per perspective.
PERSPECTIVES: Mapping[str, str] = {
    "single": "The guest is travelling alone.",
    "couple": "The guest is travelling with a partner.",
    "group": "The guest is travelling with a group or family.",
}

VISION_INSTRUCTION = (
    "You help travellers evaluate accommodation listings. A guest is viewing "
    "this {label} image to decide whether the accommodation meets their needs.\n\n"
    "Generate {n} practical, decision-relevant questions a traveller would ask "
    "about this {label}.\n\n"
    "Requirements:\n"
    "1. Focus on functionality, size, quality, or condition.\n"
    "2. Each question should help the guest decide whether to book.\n"
    "3. Use natural language.\n"
    "4. Be specific to what is visible in the image.\n"
    "5. Each question should address a different aspect.\n\n"
    "Avoid questions already answered by the photo ('Is this a hotel bathroom?') "
    "or true of any room ('Does it include a toilet?').\n\n"
    "Return ONLY the {n} questions, numbered 1-{n}, one per line."
)


def build_vision_prompt(label: str, num_questions: int = 3, perspective: str = "single") -> str:
    """Assemble the instruction sent alongside the cropped image."""
    if perspective not in PERSPECTIVES:
        available = ", ".join(sorted(PERSPECTIVES))
        raise KeyError(f"Unknown perspective {perspective!r}. Available: {available}")
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")

    parts = [VISION_INSTRUCTION.format(label=label, n=num_questions)]
    parts.append(PERSPECTIVES[perspective])
    return "\n\n".join(parts)


def pad_box(
    box: Sequence[float],
    image_size: Tuple[int, int],
    pad_frac: float = 0.35,
) -> Tuple[int, int, int, int]:
    """Expand a detection box by `pad_frac` on each side, clipped to the image.

    A tight crop of a kettle is just a kettle; the surrounding context is what
    lets the model ask about counter space or reachability. 0.35 is the value
    used in the thesis runs.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    width, height = image_size
    pad_x = pad_frac * (x2 - x1)
    pad_y = pad_frac * (y2 - y1)
    return (
        max(int(x1 - pad_x), 0),
        max(int(y1 - pad_y), 0),
        min(int(x2 + pad_x), int(width)),
        min(int(y2 + pad_y), int(height)),
    )


def parse_numbered_questions(text: str, num_questions: int, label: str) -> List[str]:
    """Parse a numbered model response into exactly `num_questions` questions.

    Models return "1. ...", "1) ...", "Q1: ...", quoted, or unquoted, and
    occasionally return fewer lines than asked for. Short remnants (<= 10
    characters) are dropped as parse noise. Any shortfall is padded with
    generic fallbacks tagged so `question_metrics.fallback_rate` can find them.
    """
    questions: List[str] = []

    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line.lstrip("0123456789").lstrip(".):").strip()
        for marker in (f"Q{i}:" for i in range(1, num_questions + 1)):
            line = line.replace(marker, "")
        line = line.strip().strip('"').strip("'").strip()
        if len(line) <= 10:
            continue
        if not line.endswith("?"):
            line += "?"
        questions.append(line)

    fallbacks = [
        f"What is the condition of the {label}?",
        f"Does the {label} appear modern and well-maintained?",
        f"What features does the {label} area include?",
        f"Is the {label} area spacious?",
        f"Would this {label} meet your accommodation needs?",
    ]
    index = 0
    while len(questions) < num_questions:
        questions.append(fallbacks[index % len(fallbacks)])
        index += 1

    return questions[:num_questions]
