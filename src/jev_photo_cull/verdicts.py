"""Normalized answers from a decision backend, independent of the wire format."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityVerdict:
    label: str
    probabilities: dict[str, float]
    # For a model verdict this is probabilities[choice], not the API confidence field.
    confidence: float
    reasons: list[str] = field(default_factory=list)
    abstained: bool = False
    confidence_raw: float | None = None


@dataclass
class KeeperVerdict:
    """``keeper_path`` is None when the backend abstains."""

    keeper_path: str | None
    probabilities: dict[str, float]
    confidence: float
    abstained: bool
    reasons: list[str] = field(default_factory=list)
    needs_review: bool = False
    confidence_raw: float | None = None
