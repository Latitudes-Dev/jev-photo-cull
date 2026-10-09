"""Typed decision questions shared by the Clef and local backends.

Workers AI Clef schema (fetched 2026-10-09):
https://developers.cloudflare.com/workers-ai/models/clef-flash/schema-input.json

A choice question needs ``type``, ``instructions``, and ``criteria`` (option id
to description, 2–255 options). Up to 64 questions and 4 images per request.
Image items are a data URL or ``{"content_type", "base64"}``. Remote URLs are
rejected. This module builds that body; it does not talk to the network.
"""

from __future__ import annotations

import base64
from typing import Any

from jev_photo_cull.heuristics import LABELS
from jev_photo_cull.records import ImageRecord

QUALITY_CRITERIA: dict[str, str] = {
    "clean": "Sharp enough and reasonably exposed to keep as a usable photo.",
    "blurry": "Motion blur or defocus has wiped out fine detail.",
    "too_dark": "Underexposed; the subject is lost in shadow.",
    "overexposed": "Highlights are blown out or the frame is washed white.",
    "unknown": "Cannot tell (unreadable, not a photograph, or genuinely ambiguous).",
}

MAX_IMAGES_PER_REQUEST = 4


def encode_image(jpeg: bytes, image_format: str) -> str | dict[str, str]:
    """Encode a JPEG for a decision request.

    ``content-type`` matches the Workers AI Clef schema object form.
    ``data-url`` matches System One's data-URL string form.
    ``raw`` matches Ollama's ``/v1/systemone`` images field, which rejects data URLs.
    """
    encoded = base64.standard_b64encode(jpeg).decode("ascii")
    if image_format == "raw":
        return encoded
    if image_format == "data-url":
        return f"data:image/jpeg;base64,{encoded}"
    if image_format == "content-type":
        return {"content_type": "image/jpeg", "base64": encoded}
    raise ValueError(f"unknown image format {image_format!r}")


def quality_question(index: int, filename: str) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": (
            f"Judge only image {index + 1} ({filename}), ignoring the other images. "
            "Which quality label fits that image?"
        ),
        "criteria": dict(QUALITY_CRITERIA),
    }


def build_quality_request(
    model: str,
    images: list[ImageRecord],
    *,
    image_format: str,
) -> dict[str, Any]:
    if not images or len(images) > MAX_IMAGES_PER_REQUEST:
        raise ValueError("quality requests carry 1 to 4 images")
    payload_images = []
    order = []
    questions = {}
    for index, record in enumerate(images):
        if not record.model_jpeg:
            raise ValueError(f"{record.relative_path} has no downscaled JPEG")
        payload_images.append(encode_image(record.model_jpeg, image_format))
        order.append({"index": index + 1, "file": record.relative_path})
        questions[f"quality_{index}"] = quality_question(index, record.relative_path)
    return {
        "model": model,
        "state": {
            "task": "Rate each attached photo. Images are listed in order and downscaled.",
            "images": order,
        },
        "images": payload_images,
        "questions": questions,
    }


def build_keeper_request(
    model: str,
    images: list[ImageRecord],
    *,
    image_format: str,
    omitted: list[str] | None = None,
) -> dict[str, Any]:
    if not images or len(images) > MAX_IMAGES_PER_REQUEST:
        raise ValueError("keeper requests carry 1 to 4 images")
    criteria: dict[str, str] = {}
    order = []
    payload_images = []
    for index, record in enumerate(images):
        if not record.model_jpeg:
            raise ValueError(f"{record.relative_path} has no downscaled JPEG")
        option = f"shot_{index + 1}"
        criteria[option] = f"Keep {record.relative_path}."
        order.append({"option": option, "file": record.relative_path})
        payload_images.append(encode_image(record.model_jpeg, image_format))
    criteria["abstain"] = "No clear winner; a person should choose."
    note = "Choose the single best frame: sharp, well exposed, and complete."
    if omitted:
        note += " Other group members were weaker on local metrics and are not shown: " + ", ".join(omitted)
    return {
        "model": model,
        "state": {
            "task": note,
            "images": order,
        },
        "images": payload_images,
        "questions": {
            "keeper": {
                "type": "choice",
                "instructions": (
                    "Which one of these photos should be kept? "
                    "Prefer the sharpest, best-exposed frame. "
                    "Choose abstain when they are too close to call."
                ),
                "criteria": criteria,
            }
        },
    }


def parse_choice(answer: object, *, expected_options: set[str]) -> tuple[str, float, dict[str, float], bool]:
    """Return choice, confidence, probabilities, and whether the model abstained."""
    if not isinstance(answer, dict):
        raise ValueError("decision answer was not an object")
    if answer.get("type") not in (None, "choice"):
        raise ValueError(f"expected a choice answer, got {answer.get('type')!r}")
    raw_probs = answer.get("probabilities")
    if not isinstance(raw_probs, dict) or not raw_probs:
        raise ValueError("choice answer is missing probabilities")
    probabilities = {str(key): float(value) for key, value in raw_probs.items()}
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in probabilities:
        raise ValueError("choice answer has no known choice")
    if expected_options and choice not in expected_options and choice not in probabilities:
        raise ValueError(f"unexpected choice {choice!r}")
    confidence = answer.get("confidence")
    if confidence is None:
        confidence = probabilities[choice]
    confidence_value = float(confidence)
    unknown_probability = answer.get("unknown_probability")
    abstained = bool(answer.get("abstained")) or choice == "abstain"
    if isinstance(unknown_probability, (int, float)) and float(unknown_probability) >= 0.5:
        abstained = True
    return choice, confidence_value, probabilities, abstained


def options_for_labels() -> set[str]:
    return set(LABELS)
