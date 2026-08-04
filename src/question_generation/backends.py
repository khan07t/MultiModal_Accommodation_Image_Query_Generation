"""Generation backends for amenity questions.

The thesis compared local seq2seq models (Flan-T5) against hosted APIs (Groq,
Gemini). Each backend exposes the same `generate(prompts) -> list[str]`
interface so `generate.py` does not care which is in use.

API keys are read from the environment. Never hard-code them.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence

FALLBACK_MARKER = "[FALLBACK]"


class GenerationBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, prompts: Sequence[str]) -> List[str]:
        """Return one generated question per prompt."""

    @staticmethod
    def _require_env(var: str) -> str:
        value = os.environ.get(var)
        if not value:
            raise EnvironmentError(
                f"{var} is not set. Export it before running, e.g.\n"
                f"    export {var}=...   (macOS/Linux)\n"
                f"    setx {var} ...     (Windows)"
            )
        return value


class FlanT5Backend(GenerationBackend):
    """Local Flan-T5 seq2seq generation."""

    def __init__(self, model_name: str = "google/flan-t5-base", batch_size: int = 8):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self.name = model_name.split("/")[-1]
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._torch = torch

    def generate(self, prompts: Sequence[str]) -> List[str]:
        outputs: List[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = list(prompts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            with self._torch.no_grad():
                generated = self.model.generate(
                    **encoded, max_new_tokens=48, num_beams=4, early_stopping=True
                )
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return [text.strip() for text in outputs]


class GroqBackend(GenerationBackend):
    """Hosted Llama / Mixtral models via the Groq API."""

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        from groq import Groq

        self.name = model
        self.client = Groq(api_key=self._require_env("GROQ_API_KEY"))

    def generate(self, prompts: Sequence[str]) -> List[str]:
        results: List[str] = []
        for prompt in prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=48,
                    temperature=0.7,
                )
                results.append(response.choices[0].message.content.strip())
            except Exception as exc:  # noqa: BLE001 - record and continue
                results.append(f"{FALLBACK_MARKER} {type(exc).__name__}")
        return results


class GeminiBackend(GenerationBackend):
    """Hosted Gemini models via the Google Generative AI API."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        import google.generativeai as genai

        genai.configure(api_key=self._require_env("GEMINI_API_KEY"))
        self.name = model
        self.model = genai.GenerativeModel(model)

    def generate(self, prompts: Sequence[str]) -> List[str]:
        results: List[str] = []
        for prompt in prompts:
            try:
                response = self.model.generate_content(prompt)
                results.append(response.text.strip())
            except Exception as exc:  # noqa: BLE001 - record and continue
                results.append(f"{FALLBACK_MARKER} {type(exc).__name__}")
        return results


class GeminiVisionBackend(GenerationBackend):
    """Gemini conditioned on the cropped detection region, not a text summary.

    This is the backend that answers Finding 3. The text-only backends receive
    `label + confidence + prompt_type`, which barely varies between rows, so
    they emit the same handful of sentences. Here the model receives the actual
    padded crop, so the questions track what is in the photograph.

    `generate()` is unavailable, because a prompt alone carries no image. Use
    `generate_for_detection()`, or `generate.py --backend gemini-vision`.
    """

    def __init__(self, model: str = "gemini-2.5-flash", pad_frac: float = 0.35):
        import google.generativeai as genai

        genai.configure(api_key=self._require_env("GEMINI_API_KEY"))
        self.name = model
        self.pad_frac = pad_frac
        self.model = genai.GenerativeModel(model)

    def generate(self, prompts: Sequence[str]) -> List[str]:
        raise NotImplementedError(
            "GeminiVisionBackend needs an image. Call generate_for_detection(...) "
            "with an image path and a bounding box."
        )

    def generate_for_detection(
        self,
        image_path: "str | Path",
        box: Sequence[float],
        label: str,
        num_questions: int = 3,
        perspective: str = "single",
    ) -> List[str]:
        """Return `num_questions` questions about one detected amenity."""
        from PIL import Image

        from .prompts import build_vision_prompt, pad_box, parse_numbered_questions

        prompt = build_vision_prompt(label, num_questions, perspective)

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            crop = image.crop(pad_box(box, image.size, self.pad_frac))
            response = self.model.generate_content([prompt, crop])
            return parse_numbered_questions(response.text, num_questions, label)
        except Exception as exc:  # noqa: BLE001 - record and continue
            marker = f"{FALLBACK_MARKER} {type(exc).__name__}"
            return [marker] * num_questions


BACKENDS = {
    "flan-t5-base": lambda: FlanT5Backend("google/flan-t5-base"),
    "flan-t5-large": lambda: FlanT5Backend("google/flan-t5-large"),
    "llama-8b": lambda: GroqBackend("llama-3.1-8b-instant"),
    "llama-70b": lambda: GroqBackend("llama-3.3-70b-versatile"),
    "gemini-flash": lambda: GeminiBackend("gemini-2.5-flash"),
    "gemini-pro": lambda: GeminiBackend("gemini-2.5-pro"),
    "gemini-vision": lambda: GeminiVisionBackend("gemini-2.5-flash"),
}

VISION_BACKENDS = {"gemini-vision"}


def get_backend(key: str) -> GenerationBackend:
    if key not in BACKENDS:
        raise KeyError(f"Unknown backend {key!r}. Available: {', '.join(BACKENDS)}")
    return BACKENDS[key]()
