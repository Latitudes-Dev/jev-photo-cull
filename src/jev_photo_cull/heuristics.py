"""Turn exposure and sharpness measurements into a single quality label and a keeper score."""

from __future__ import annotations

import math
from dataclasses import dataclass

from jev_photo_cull.records import ImageRecord

LABELS = ("clean", "blurry", "too_dark", "overexposed", "unknown")
DEFECTS = ("blurry", "too_dark", "overexposed")


@dataclass(frozen=True)
class HeuristicConfig:
    """Thresholds tuned for the synthetic fixtures and typical downscaled photos.

    Sharpness is the variance of a Laplacian after a 1st/99th percentile stretch,
    measured on a long edge of at most 1024 px. Exposure uses mean luminance
    (0–255) plus the fraction of pixels clipped near black or white.
    """

    blur_sharpness: float = 80.0
    dark_mean: float = 48.0
    dark_mean_soft: float = 72.0
    dark_shadow_fraction: float = 0.50
    bright_mean: float = 205.0
    bright_mean_soft: float = 185.0
    bright_highlight_fraction: float = 0.40
    keeper_margin: float = 0.06


@dataclass(frozen=True)
class HeuristicJudgement:
    label: str
    confidence: float
    probabilities: dict[str, float]
    reasons: list[str]
    quality_score: float


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _spread(label: str, confidence: float) -> dict[str, float]:
    """Put ``confidence`` on the chosen label and share the rest across the others."""
    confidence = _clamp(confidence, 0.05, 0.99)
    others = [name for name in LABELS if name != label]
    remainder = (1.0 - confidence) / len(others)
    return {name: (confidence if name == label else remainder) for name in LABELS}


def _blur_confidence(sharpness: float, threshold: float) -> float:
    if sharpness >= threshold:
        return _clamp(0.55 + 0.4 * (sharpness - threshold) / (threshold * 4))
    return _clamp(0.55 + 0.4 * (threshold - sharpness) / threshold)


def _exposure_confidence(distance: float, scale: float) -> float:
    return _clamp(0.55 + 0.4 * (distance / scale))


def judge(record: ImageRecord, config: HeuristicConfig | None = None) -> HeuristicJudgement:
    settings = config or HeuristicConfig()
    if not record.readable or record.sharpness is None or record.mean_luminance is None:
        reason = record.error or "image could not be measured"
        return HeuristicJudgement(
            label="unknown",
            confidence=0.9 if record.error else 0.4,
            probabilities=_spread("unknown", 0.9 if record.error else 0.4),
            reasons=[reason],
            quality_score=0.0,
        )

    mean = record.mean_luminance
    shadow = record.shadow_fraction or 0.0
    highlight = record.highlight_fraction or 0.0
    sharpness = record.sharpness
    reasons: list[str] = [
        f"sharpness {sharpness:.1f} (blur below {settings.blur_sharpness:.0f})",
        f"mean luminance {mean:.1f}, shadows {shadow:.0%}, highlights {highlight:.0%}",
    ]

    candidates: list[tuple[str, float, str]] = []
    if mean >= settings.bright_mean or (
        mean >= settings.bright_mean_soft and highlight >= settings.bright_highlight_fraction
    ):
        distance = max(mean - settings.bright_mean_soft, highlight * 80.0)
        candidates.append(
            (
                "overexposed",
                _exposure_confidence(distance, 50.0),
                "highlights are clipped or the frame is very bright",
            )
        )
    if mean <= settings.dark_mean or (
        mean <= settings.dark_mean_soft and shadow >= settings.dark_shadow_fraction
    ):
        distance = max(settings.dark_mean_soft - mean, shadow * 80.0)
        candidates.append(
            (
                "too_dark",
                _exposure_confidence(distance, 50.0),
                "the frame is underexposed or mostly crushed shadow",
            )
        )
    if sharpness < settings.blur_sharpness and not candidates:
        # Exposure defects already explain a flat histogram. Only call the
        # remainder blurry when the stretch still cannot find edges.
        candidates.append(
            (
                "blurry",
                _blur_confidence(sharpness, settings.blur_sharpness),
                "Laplacian variance is below the blur threshold after contrast stretch",
            )
        )
    elif sharpness < settings.blur_sharpness * 0.45 and candidates:
        candidates.append(
            (
                "blurry",
                _blur_confidence(sharpness, settings.blur_sharpness),
                "the frame is also soft after contrast stretch",
            )
        )

    score = quality_score(record, settings)
    if not candidates:
        confidence = _blur_confidence(sharpness, settings.blur_sharpness)
        reasons.append("exposure is in range and edges survive the blur threshold")
        return HeuristicJudgement(
            label="clean",
            confidence=confidence,
            probabilities=_spread("clean", confidence),
            reasons=reasons,
            quality_score=score,
        )

    candidates.sort(key=lambda item: item[1], reverse=True)
    label, confidence, why = candidates[0]
    reasons.append(why)
    if len(candidates) > 1 and candidates[0][1] - candidates[1][1] < 0.08:
        reasons.append(
            f"also consistent with {candidates[1][0]} ({candidates[1][2]})"
        )
    return HeuristicJudgement(
        label=label,
        confidence=confidence,
        probabilities=_spread(label, confidence),
        reasons=reasons,
        quality_score=score,
    )


def quality_score(record: ImageRecord, config: HeuristicConfig | None = None) -> float:
    """Higher is a better keeper. Unreadable files score 0."""
    settings = config or HeuristicConfig()
    if not record.readable or record.sharpness is None or record.mean_luminance is None:
        return 0.0
    sharp_term = record.sharpness / (record.sharpness + settings.blur_sharpness)
    mean_term = 1.0 - min(1.0, abs(record.mean_luminance - 135.0) / 135.0)
    clip = (record.shadow_fraction or 0.0) + (record.highlight_fraction or 0.0)
    clip_term = 1.0 - min(1.0, clip)
    return 0.55 * sharp_term + 0.30 * mean_term + 0.15 * clip_term


def softmax(scores: list[float], temperature: float = 4.0) -> list[float]:
    if not scores:
        return []
    peak = max(scores)
    weights = [math.exp((score - peak) * temperature) for score in scores]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]
