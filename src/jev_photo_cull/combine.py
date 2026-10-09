"""Merge pass-1 measurements with an optional decision-model answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from jev_photo_cull.heuristics import DEFECTS
from jev_photo_cull.verdicts import KeeperVerdict, QualityVerdict


@dataclass
class CombinedQuality:
    label: str
    confidence: float
    probabilities: dict[str, float]
    needs_review: bool
    reasons: list[str] = field(default_factory=list)


def combine_quality(
    heuristic: QualityVerdict,
    model: QualityVerdict | None,
    *,
    review_below: float,
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
    reasons.append(
        f"disagreement: heuristics={heuristic.label} ({heuristic.confidence:.2f}), "
        f"model={model.label} ({model.confidence:.2f})"
    )
    if model.confidence >= heuristic.confidence:
        label, confidence, probabilities = model.label, model.confidence, model.probabilities
        reasons.append("using the model label and flagging the photo for review")
    else:
        label, confidence, probabilities = heuristic.label, heuristic.confidence, heuristic.probabilities
        reasons.append("using the heuristic label and flagging the photo for review")
    return CombinedQuality(
        label=label,
        confidence=confidence,
        probabilities=dict(probabilities),
        needs_review=True,
        reasons=reasons,
    )


def combine_keeper(
    heuristic: KeeperVerdict | None,
    model: KeeperVerdict | None,
    *,
    review_below: float,
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
        )
    reasons.append(
        f"keeper disagreement: heuristics={heuristic.keeper_path}, model={model.keeper_path}"
    )
    winner = model if model.confidence >= heuristic.confidence else heuristic
    return KeeperVerdict(
        keeper_path=winner.keeper_path,
        probabilities=winner.probabilities,
        confidence=winner.confidence,
        abstained=False,
        reasons=reasons,
        needs_review=True,
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
