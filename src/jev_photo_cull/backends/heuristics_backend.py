"""Pass-2 backend that only restates pass-1 measurements. No network."""

from __future__ import annotations

from jev_photo_cull.analyze import hamming
from jev_photo_cull.grouping import GroupingConfig
from jev_photo_cull.heuristics import HeuristicConfig, judge, quality_score, softmax
from jev_photo_cull.records import Group, ImageRecord
from jev_photo_cull.verdicts import KeeperVerdict, QualityVerdict


def _all_near(members: list[ImageRecord], near_hamming: int) -> bool:
    hashes = [member.perceptual_hash for member in members]
    if any(value is None for value in hashes):
        return False
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            left_hash = hashes[left]
            right_hash = hashes[right]
            if left_hash is None or right_hash is None or hamming(left_hash, right_hash) > near_hamming:
                return False
    return True


class HeuristicsBackend:
    name = "heuristics"
    model_id: str | None = None

    def __init__(
        self,
        heuristic_config: HeuristicConfig | None = None,
        grouping_config: GroupingConfig | None = None,
        review_below: float = 0.60,
    ) -> None:
        self.heuristic_config = heuristic_config or HeuristicConfig()
        self.grouping_config = grouping_config or GroupingConfig()
        self.review_below = review_below

    def classify(self, records: list[ImageRecord]) -> dict[str, QualityVerdict]:
        verdicts: dict[str, QualityVerdict] = {}
        for record in records:
            judgement = judge(record, self.heuristic_config)
            verdicts[record.relative_path] = QualityVerdict(
                label=judgement.label,
                probabilities=judgement.probabilities,
                confidence=judgement.confidence,
                reasons=list(judgement.reasons),
            )
        return verdicts

    def choose_keepers(self, groups: list[Group]) -> dict[str, KeeperVerdict]:
        verdicts: dict[str, KeeperVerdict] = {}
        for group in groups:
            if group.size < 2:
                continue
            verdicts[group.id] = self._choose(group)
        return verdicts

    def _choose(self, group: Group) -> KeeperVerdict:
        readable = [member for member in group.members if member.readable]
        if len(readable) < 2:
            return KeeperVerdict(
                keeper_path=None,
                probabilities={"abstain": 1.0},
                confidence=0.2,
                abstained=True,
                reasons=["not enough readable frames to pick a keeper"],
                needs_review=True,
            )
        scores = [quality_score(member, self.heuristic_config) for member in readable]
        probabilities = softmax(scores)
        ranked = sorted(zip(readable, scores, probabilities), key=lambda item: (-item[1], item[0].relative_path))
        best, best_score, best_probability = ranked[0]
        second, second_score, _second_probability = ranked[1]
        gap = best_score - second_score
        named = {member.relative_path: float(probability) for member, _score, probability in ranked}
        if gap < self.heuristic_config.keeper_margin:
            tied = [member for member, score, _probability in ranked if best_score - score < self.heuristic_config.keeper_margin]
            equivalent = _all_near(tied, self.grouping_config.near_hamming)
            if equivalent:
                keeper = min(tied, key=lambda member: member.relative_path)
                return KeeperVerdict(
                    keeper_path=keeper.relative_path,
                    probabilities=named,
                    confidence=0.72,
                    abstained=False,
                    reasons=[
                        "top frames are equivalent on sharpness and exposure; "
                        f"keeping {keeper.relative_path} (first path among the tie)"
                    ],
                )
            return KeeperVerdict(
                keeper_path=None,
                probabilities={**named, "abstain": max(0.5, 1.0 - gap)},
                confidence=max(0.5, 1.0 - gap * 4),
                abstained=True,
                reasons=[
                    f"top two quality scores differ by {gap:.3f}, under the {self.heuristic_config.keeper_margin:.3f} margin"
                ],
                needs_review=True,
            )
        confidence = min(0.95, 0.55 + gap * 2)
        return KeeperVerdict(
            keeper_path=best.relative_path,
            probabilities=named,
            confidence=confidence,
            abstained=False,
            reasons=[
                f"highest local quality score {best_score:.3f} "
                f"(next {second_score:.3f}, probability {best_probability:.2f})"
            ],
            needs_review=confidence < self.review_below,
        )
