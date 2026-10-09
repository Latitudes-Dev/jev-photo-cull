"""In-memory records for one scanned image and one duplicate/burst group."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ImageRecord:
    """One file from the input tree. JPEG payloads are derived copies, never written back."""

    relative_path: str
    absolute_path: str
    sha256: str
    file_size: int
    perceptual_hash: int | None = None
    width: int | None = None
    height: int | None = None
    captured_at: datetime | None = None
    mean_luminance: float | None = None
    luminance_std: float | None = None
    shadow_fraction: float | None = None
    highlight_fraction: float | None = None
    rgb_mean: tuple[float, float, float] | None = None
    histogram: list[int] | None = None
    sharpness: float | None = None
    error: str | None = None
    thumb_jpeg: bytes | None = None
    model_jpeg: bytes | None = None

    @property
    def readable(self) -> bool:
        return self.error is None and self.perceptual_hash is not None

    @property
    def perceptual_hash_hex(self) -> str | None:
        if self.perceptual_hash is None:
            return None
        return f"dhash:{self.perceptual_hash:016x}"


@dataclass
class Group:
    id: str
    members: list[ImageRecord]
    kind: str
    edge_kinds: set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.members)
