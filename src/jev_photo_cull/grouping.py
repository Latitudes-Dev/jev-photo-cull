"""Group exact copies, near-duplicates, and time-proximate bursts.

Flat frames (blank walls, blown skies) share a dHash of zero. A blank pairs
only with another blank of a similar color, and a near-duplicate link also
requires a similar mean luminance so a dark frame is not glued to a bright one.
Bursts may span a wider exposure gap when EXIF timestamps are close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jev_photo_cull.analyze import hamming
from jev_photo_cull.records import Group, ImageRecord


@dataclass(frozen=True)
class GroupingConfig:
    near_hamming: int = 8
    burst_hamming: int = 16
    burst_seconds: float = 2.0
    flat_std: float = 10.0
    flat_color_distance: float = 28.0
    # Near-duplicates must also be in a similar exposure. A blank frame hashes
    # like other blank frames, so blanks only pair with blanks of a similar color.
    near_mean_delta: float = 35.0
    burst_mean_delta: float = 160.0
    blank_sharpness: float = 1.0


class _UnionFind:
    def __init__(self, count: int) -> None:
        self._parent = list(range(count))
        self._rank = [0] * count

    def find(self, index: int) -> int:
        parent = self._parent[index]
        if parent != index:
            self._parent[index] = self.find(parent)
        return self._parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1


def _color_distance(left: ImageRecord, right: ImageRecord) -> float:
    if left.rgb_mean is None or right.rgb_mean is None:
        if left.mean_luminance is None or right.mean_luminance is None:
            return 255.0
        return abs(left.mean_luminance - right.mean_luminance)
    return math.dist(left.rgb_mean, right.rgb_mean)


def _blank(record: ImageRecord, config: GroupingConfig) -> bool:
    sharpness = record.sharpness or 0.0
    spread = record.luminance_std or 0.0
    return sharpness < config.blank_sharpness and spread <= config.flat_std


def _mean_gap(left: ImageRecord, right: ImageRecord) -> float:
    if left.mean_luminance is None or right.mean_luminance is None:
        return 255.0
    return abs(left.mean_luminance - right.mean_luminance)


def link_kind(
    left: ImageRecord,
    right: ImageRecord,
    config: GroupingConfig,
) -> str | None:
    """Return ``identical``, ``near``, ``burst``, or None if the pair should stay apart."""
    if not left.readable or not right.readable:
        return None
    if left.sha256 == right.sha256:
        return "identical"
    assert left.perceptual_hash is not None and right.perceptual_hash is not None
    distance = hamming(left.perceptual_hash, right.perceptual_hash)
    left_blank = _blank(left, config)
    right_blank = _blank(right, config)
    if left_blank or right_blank:
        # A featureless frame collides in dHash with every other featureless frame.
        if left_blank != right_blank:
            return None
        if _color_distance(left, right) > config.flat_color_distance:
            return None
    mean_gap = _mean_gap(left, right)
    if distance <= config.near_hamming and mean_gap <= config.near_mean_delta:
        return "near"
    if (
        left.captured_at is not None
        and right.captured_at is not None
        and distance <= config.burst_hamming
        and mean_gap <= config.burst_mean_delta
    ):
        delta = abs((left.captured_at - right.captured_at).total_seconds())
        if delta <= config.burst_seconds:
            return "burst"
    return None


def _group_kind(size: int, kinds: set[str]) -> str:
    if size == 1:
        return "single"
    if kinds <= {"burst"}:
        return "burst"
    if "burst" in kinds:
        return "mixed"
    if kinds <= {"identical"}:
        return "identical"
    return "near-duplicate"


def group_records(records: list[ImageRecord], config: GroupingConfig | None = None) -> list[Group]:
    """Cluster records. Group ids follow the earliest relative path in each cluster."""
    settings = config or GroupingConfig()
    count = len(records)
    union = _UnionFind(count)
    for left in range(count):
        for right in range(left + 1, count):
            if link_kind(records[left], records[right], settings) is not None:
                union.union(left, right)

    clusters: dict[int, list[ImageRecord]] = {}
    for index, record in enumerate(records):
        clusters.setdefault(union.find(index), []).append(record)

    def sort_key(members: list[ImageRecord]) -> str:
        return min(member.relative_path for member in members)

    groups: list[Group] = []
    for number, members in enumerate(sorted(clusters.values(), key=sort_key), start=1):
        members.sort(key=lambda record: record.relative_path)
        kinds: set[str] = set()
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                kind = link_kind(left, right, settings)
                if kind:
                    kinds.add(kind)
        groups.append(
            Group(
                id=f"g{number:04d}",
                members=members,
                kind=_group_kind(len(members), kinds),
                edge_kinds=kinds,
            )
        )
    return groups
