"""Merge pass-1 measurements with an optional decision-model answer.

The heuristic score and the model score are both probabilities in ``[0, 1]``,
but they are not calibrated to each other. On a live Clef-flash run the
chosen-class probability had a median of about 0.83, while the local
heuristic confidence on ordinary photos is about 0.89. ``DEFAULT_MODEL_MARGIN``
is added to the model score before the comparison so a clearly more certain
model can override the heuristic.

A gap of ``CLOSE_CALL_GAP`` or less keeps the higher adjusted score as the
suggested label and sets ``needs_review``. A wider gap uses the higher score
without forcing a review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jev_photo_cull.heuristics import DEFECTS
from jev_photo_cull.verdicts import KeeperVerdict, QualityVerdict

# Added to probabilities[choice] before comparing with heuristic confidence.
DEFAULT_MODEL_MARGIN = 0.10
# |adjusted model score - heuristic confidence| at or under this is a close call.
CLOSE_CALL_GAP = 0.05


@dataclass
class CombinedQuality:
    label: str
    confidence: float
    probabilities: dict[str, float]
    needs_review: bool
    reasons: list[str] = field(default_factory=list)


def _adjusted_gap(model_score: float, heuristic_score: float, model_margin: float) -> float:
    """Model score plus margin, minus the heuristic score.

    Rounded so a calibration tie such as ``0.70 + 0.10`` against ``0.80``
    compares as zero instead of a tiny negative float.
    """
    return round(model_score + model_margin - heuristic_score, 6)


def combine_quality(
    heuristic: QualityVerdict,
    model: QualityVerdict | None,
    *,
    review_below: float,
    model_margin: float = DEFAULT_MODEL_MARGIN,
) -> CombinedQuality:
    reasons = list(heuristic.reasons)
    if model is None:
        needs = heuristic.label == "unknown" or heuristic.confidence < review_below or heuristic.abstained
        return CombinedQuality(
            label=heuristic.label,
            confidence=heuristic.confidence,
            probabilities=dict(heuristic.probabilities),
            needs_review=needs,
            reasons=reasons,
        )
    reasons.extend(model.reasons)
    if heuristic.label == model.label:
        confidence = min(0.99, (heuristic.confidence + model.confidence) / 2 + 0.08)
        needs = model.label == "unknown" or confidence < review_below or model.abstained
        reasons.append(f"heuristics and the model agree on {model.label}")
        return CombinedQuality(
            label=model.label,
            confidence=confidence,
            probabilities=dict(model.probabilities),
            needs_review=needs,
            reasons=reasons,
        )
    adjusted = model.confidence + model_margin
    gap = _adjusted_gap(model.confidence, heuristic.confidence, model_margin)
    reasons.append(
        f"disagreement: heuristics={heuristic.label} ({heuristic.confidence:.2f}), "
        f"model={model.label} (p={model.confidence:.2f}, adjusted {adjusted:.2f})"
    )
    close = abs(gap) <= CLOSE_CALL_GAP
    if gap >= 0:
        label, confidence, probabilities = model.label, model.confidence, model.probabilities
        source = model
        if close:
            reasons.append("close call; using the model label and flagging the photo for review")
        else:
            reasons.append("model chosen-class probability is clearly higher; using the model label")
    else:
        label, confidence, probabilities = heuristic.label, heuristic.confidence, heuristic.probabilities
        source = heuristic
        if close:
            reasons.append("close call; using the heuristic label and flagging the photo for review")
        else:
            reasons.append("heuristic confidence is clearly higher; using the heuristic label")
    needs_review = close or source.label == "unknown" or source.abstained or source.confidence < review_below
    return CombinedQuality(
        label=label,
        confidence=confidence,
        probabilities=dict(probabilities),
        needs_review=needs_review,
        reasons=reasons,
    )


def combine_keeper(
    heuristic: KeeperVerdict | None,
    model: KeeperVerdict | None,
    *,
    review_below: float,
    model_margin: float = DEFAULT_MODEL_MARGIN,
) -> KeeperVerdict | None:
    """Combine keeper picks for a multi-image group. None means the group is a singleton."""
    if heuristic is None and model is None:
        return None
    if model is None:
        return heuristic
    if heuristic is None:
        return model
    reasons = list(heuristic.reasons) + list(model.reasons)
    if heuristic.abstained and model.abstained:
        return KeeperVerdict(
            keeper_path=None,
            probabilities=model.probabilities or heuristic.probabilities,
            confidence=min(heuristic.confidence, model.confidence),
            abstained=True,
            reasons=reasons + ["both passes abstained"],
            needs_review=True,
            confidence_raw=model.confidence_raw,
        )
    if model.abstained and not heuristic.abstained:
        use_heuristic = heuristic.confidence >= review_below
        return KeeperVerdict(
            keeper_path=heuristic.keeper_path if use_heuristic else None,
            probabilities=heuristic.probabilities,
            confidence=heuristic.confidence,
            abstained=not use_heuristic,
            reasons=reasons + ["model abstained; falling back to local scores"],
            needs_review=True,
            confidence_raw=heuristic.confidence_raw,
        )
    if heuristic.abstained and not model.abstained:
        use_model = model.confidence >= review_below
        return KeeperVerdict(
            keeper_path=model.keeper_path if use_model else None,
            probabilities=model.probabilities,
            confidence=model.confidence,
            abstained=not use_model,
            reasons=reasons + ["local scores abstained; using the model"],
            needs_review=True,
            confidence_raw=model.confidence_raw,
        )
    assert heuristic.keeper_path is not None and model.keeper_path is not None
    if heuristic.keeper_path == model.keeper_path:
        confidence = min(0.99, max(heuristic.confidence, model.confidence))
        return KeeperVerdict(
            keeper_path=heuristic.keeper_path,
            probabilities=model.probabilities,
            confidence=confidence,
            abstained=False,
            reasons=reasons + [f"both passes keep {heuristic.keeper_path}"],
            needs_review=confidence < review_below,
            confidence_raw=model.confidence_raw,
        )
    adjusted = model.confidence + model_margin
    gap = _adjusted_gap(model.confidence, heuristic.confidence, model_margin)
    close = abs(gap) <= CLOSE_CALL_GAP
    reasons.append(
        f"keeper disagreement: heuristics={heuristic.keeper_path} ({heuristic.confidence:.2f}), "
        f"model={model.keeper_path} (p={model.confidence:.2f}, adjusted {adjusted:.2f})"
    )
    winner = model if gap >= 0 else heuristic
    if close:
        reasons.append("close call; flagging the group for review")
    return KeeperVerdict(
        keeper_path=winner.keeper_path,
        probabilities=winner.probabilities,
        confidence=winner.confidence,
        abstained=False,
        reasons=reasons,
        needs_review=close or winner.confidence < review_below,
        confidence_raw=winner.confidence_raw,
    )


def singleton_keep(label: str, needs_review: bool) -> bool | None:
    """A lone photo is kept when it looks clean, dropped when a defect is confident."""
    if needs_review or label == "unknown":
        return None
    if label == "clean":
        return True
    if label in DEFECTS:
        return False
    return None
