"""Question generation with Gemini on cropped amenity regions.

The thesis runs used Vertex AI. Set GOOGLE_CLOUD_PROJECT to use Vertex,
or GEMINI_API_KEY to use the public Gemini API.
"""

import io
import os
import re

from .config import GENERATION_CONFIG, PROFILES, QUESTIONS_GENERAL, QUESTIONS_PER_PROFILE
from .cropping import crop_box, safe_open_image
from .prompts import PROFILE_EXAMPLES, build_general_prompt, build_profile_prompt


def get_client():
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        return genai.Client(vertexai=True, project=project, location="global")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _call(client, model_name, prompt, crop):
    from google.genai import types

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")],
        config=types.GenerateContentConfig(**GENERATION_CONFIG),
    )
    return response.text.strip()


def parse_numbered(text: str):
    questions = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*(Q?\d+[\.\):]?)\s*", "", line.strip()).strip('"').strip("'")
        if line and not line.endswith("?"):
            line += "?"
        if len(line) > 10:
            questions.append(line)
    return questions


def general_fallback(label):
    return [
        f"What is the condition of the {label}?",
        f"Does the {label} appear modern and well-maintained?",
        f"What features does the {label} include?",
        f"Is the {label} user-friendly?",
        f"Would this {label} meet your accommodation needs?",
    ]


def generate_questions(client, model_name, image_path, bbox, label, num_questions=QUESTIONS_GENERAL):
    """Five decision-relevant questions for one detection."""
    crop = crop_box(safe_open_image(image_path), bbox)
    questions = parse_numbered(_call(client, model_name, build_general_prompt(label, num_questions), crop))
    fallback = general_fallback(label)
    while len(questions) < num_questions:
        questions.append(fallback[len(questions)])
    return questions[:num_questions]


def generate_profile_questions(client, model_name, image_path, bbox, label,
                               num_questions=QUESTIONS_PER_PROFILE):
    """Three questions each for solo, couple and group travellers, from one call."""
    crop = crop_box(safe_open_image(image_path), bbox)
    text = _call(client, model_name, build_profile_prompt(label, num_questions), crop)

    headers = {"SINGLE": "single", "COUPLE": "couple", "GROUP": "group"}
    out = {}
    for header, key in headers.items():
        match = re.search(rf"{header}:(.*?)(?={'|'.join(headers)}:|$)", text, re.DOTALL | re.IGNORECASE)
        questions = parse_numbered(match.group(1)) if match else []
        fallback = PROFILE_EXAMPLES.get(label.lower(), PROFILE_EXAMPLES["bathtub"])[key]
        while len(questions) < num_questions:
            questions.append(fallback[len(questions)] if len(questions) < len(fallback)
                             else f"What features of the {label} are important for this use case?")
        out[key] = questions[:num_questions]
    return out


assert set(PROFILES) == {"single", "couple", "group"}
