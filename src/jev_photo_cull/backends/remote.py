"""Shared batching, cache, and rate limit for HTTP decision backends."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from jev_photo_cull.cache import DecisionCache
from jev_photo_cull.decisions import (
    DECISION_SCORE_VERSION,
    MAX_IMAGES_PER_REQUEST,
    QUALITY_CRITERIA,
    build_keeper_request,
    build_quality_request,
    parse_choice,
)
from jev_photo_cull.errors import PhotoCullError
from jev_photo_cull.heuristics import quality_score
from jev_photo_cull.rate_limit import RateLimiter
from jev_photo_cull.records import Group, ImageRecord
from jev_photo_cull.verdicts import KeeperVerdict, QualityVerdict

SCHEMA_TAG = hashlib.sha256(
    json.dumps(QUALITY_CRITERIA, sort_keys=True).encode("utf-8")
).hexdigest()[:12]

PostJson = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, dict[str, str], bytes]]


class DecisionClient:
    """Posts System One style bodies and turns answers into verdicts."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        image_format: str,
        cache: DecisionCache,
        limiter: RateLimiter,
        post: PostJson,
        url: str,
        headers: dict[str, str],
        timeout: float = 120.0,
        unwrap: Callable[[int, dict[str, str], bytes], dict[str, Any]],
        review_below: float = 0.60,
        sleep: Callable[[float], None] | None = None,
        max_attempts: int = 4,
    ) -> None:
        self.name = name
        self.model = model
        self.image_format = image_format
        self.cache = cache
        self.limiter = limiter
        self.post = post
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.unwrap = unwrap
        self.review_below = review_below
        self.sleep = sleep or __import__("time").sleep
        self.max_attempts = max_attempts
        self.calls = 0
        self.neurons = 0.0
        self.input_tokens = 0

    @property
    def usage(self) -> dict[str, Any]:
        """HTTP attempts, summed ``cf-ai-neurons``, and summed ``usage.input_tokens``.

        ``usage.output_tokens`` is ignored. Clef reports 0 for it.
        Cached answers do not add to these totals.
        """
        return {
            "calls": self.calls,
            "neurons": round(self.neurons, 4),
            "input_tokens": self.input_tokens,
        }

    def classify(self, records: list[ImageRecord]) -> dict[str, QualityVerdict]:
        verdicts: dict[str, QualityVerdict] = {}
        pending: list[ImageRecord] = []
        for record in records:
            if not record.readable or not record.model_jpeg:
                verdicts[record.relative_path] = QualityVerdict(
                    label="unknown",
                    probabilities={"unknown": 1.0},
                    confidence=0.9,
                    reasons=[record.error or "no image bytes to send"],
                    abstained=True,
                )
                continue
            cached = self.cache.get(self._quality_key(record))
            if isinstance(cached, dict):
                verdicts[record.relative_path] = _verdict_from_cache(cached)
            else:
                pending.append(record)
        for chunk in _chunks(pending, MAX_IMAGES_PER_REQUEST):
            body = build_quality_request(self.model, chunk, image_format=self.image_format)
            payload = self._send(body)
            answers = _answers(payload)
            for index, record in enumerate(chunk):
                answer = answers.get(f"quality_{index}")
                parsed = parse_choice(answer, expected_options=set(QUALITY_CRITERIA))
                label = "unknown" if parsed.abstained else parsed.choice
                if label not in QUALITY_CRITERIA:
                    raise PhotoCullError(f"model returned an unknown quality label {label!r}")
                verdict = QualityVerdict(
                    label=label,
                    probabilities=parsed.probabilities,
                    confidence=parsed.score,
                    reasons=[f"{self.name} {self.model} chose {label} (p={parsed.score:.2f})"],
                    abstained=parsed.abstained,
                    confidence_raw=parsed.confidence_raw,
                )
                verdicts[record.relative_path] = verdict
                self.cache.put(self._quality_key(record), _verdict_to_cache(verdict))
        return verdicts

    def choose_keepers(self, groups: list[Group]) -> dict[str, KeeperVerdict]:
        verdicts: dict[str, KeeperVerdict] = {}
        for group in groups:
            if group.size < 2:
                continue
            readable = [member for member in group.members if member.readable and member.model_jpeg]
            if len(readable) < 2:
                verdicts[group.id] = KeeperVerdict(
                    keeper_path=None,
                    probabilities={"abstain": 1.0},
                    confidence=0.2,
                    abstained=True,
                    reasons=["not enough readable frames for the decision model"],
                    needs_review=True,
                )
                continue
            ranked = sorted(readable, key=lambda member: quality_score(member), reverse=True)
            shown = ranked[:MAX_IMAGES_PER_REQUEST]
            omitted = [member.relative_path for member in ranked[MAX_IMAGES_PER_REQUEST:]]
            cache_key = self._keeper_key(shown)
            cached = self.cache.get(cache_key)
            if isinstance(cached, dict):
                verdicts[group.id] = _keeper_from_cache(cached)
                continue
            body = build_keeper_request(
                self.model,
                shown,
                image_format=self.image_format,
                omitted=omitted,
            )
            payload = self._send(body)
            answers = _answers(payload)
            options = {f"shot_{index + 1}" for index in range(len(shown))} | {"abstain"}
            parsed = parse_choice(answers.get("keeper"), expected_options=options)
            gap = _top_gap(parsed.probabilities)
            low_margin = gap < 0.12 and parsed.choice != "abstain"
            if parsed.abstained or parsed.score < self.review_below or low_margin:
                verdict = KeeperVerdict(
                    keeper_path=None,
                    probabilities=parsed.probabilities,
                    confidence=parsed.score,
                    abstained=True,
                    reasons=[
                        f"{self.name} abstained on this group"
                        if parsed.abstained
                        else f"{self.name} keeper margin {gap:.2f} or confidence {parsed.score:.2f} is too low"
                    ],
                    needs_review=True,
                    confidence_raw=parsed.confidence_raw,
                )
            else:
                index = int(parsed.choice.split("_", 1)[1]) - 1
                keeper = shown[index]
                verdict = KeeperVerdict(
                    keeper_path=keeper.relative_path,
                    probabilities=parsed.probabilities,
                    confidence=parsed.score,
                    abstained=False,
                    reasons=[f"{self.name} keeps {keeper.relative_path} (p={parsed.score:.2f})"],
                    needs_review=parsed.score < self.review_below,
                    confidence_raw=parsed.confidence_raw,
                )
            verdicts[group.id] = verdict
            self.cache.put(cache_key, _keeper_to_cache(verdict))
        return verdicts

    def _send(self, body: dict[str, Any]) -> dict[str, Any]:
        delay = 1.0
        last_status = 0
        last_body = b""
        for attempt in range(self.max_attempts):
            self.limiter.wait()
            self.calls += 1
            status, headers, raw = self.post(self.url, self.headers, body, self.timeout)
            last_status, last_body = status, raw
            self.neurons += neurons_from_headers(headers)
            if status == 429 or status >= 500:
                if attempt + 1 >= self.max_attempts:
                    break
                retry_after = _retry_after(headers)
                self.sleep(retry_after if retry_after is not None else delay)
                delay = min(delay * 2, 30.0)
                continue
            payload = self.unwrap(status, headers, raw)
            self.input_tokens += input_tokens_from_payload(payload)
            return payload
        detail = last_body.decode("utf-8", errors="replace")[:500]
        raise PhotoCullError(
            f"{self.name} request failed after {self.max_attempts} attempts "
            f"(HTTP {last_status}): {detail}"
        )

    def _quality_key(self, record: ImageRecord) -> str:
        jpeg_hash = hashlib.sha256(record.model_jpeg or b"").hexdigest()[:16]
        return (
            f"quality|{DECISION_SCORE_VERSION}|{SCHEMA_TAG}|{self.name}|{self.model}|"
            f"{self.image_format}|{record.sha256}|{jpeg_hash}"
        )

    def _keeper_key(self, records: list[ImageRecord]) -> str:
        parts = []
        for record in records:
            jpeg_hash = hashlib.sha256(record.model_jpeg or b"").hexdigest()[:12]
            parts.append(f"{record.sha256}:{jpeg_hash}")
        return (
            f"keeper|{DECISION_SCORE_VERSION}|{SCHEMA_TAG}|{self.name}|{self.model}|"
            f"{self.image_format}|{','.join(parts)}"
        )


def _chunks(items: list[ImageRecord], size: int) -> list[list[ImageRecord]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _answers(payload: dict[str, Any]) -> dict[str, Any]:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise PhotoCullError("decision response did not include an answers object")
    return answers


def _header_value(headers: dict[str, str], name: str) -> str | None:
    folded = name.lower()
    for key, value in headers.items():
        if key.lower() == folded:
            return value
    return None


def neurons_from_headers(headers: dict[str, str]) -> float:
    """Read the Workers AI ``cf-ai-neurons`` response header, case-insensitively."""
    raw = _header_value(headers, "cf-ai-neurons")
    if raw is None or not str(raw).strip():
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def input_tokens_from_payload(payload: dict[str, Any]) -> int:
    """Sum only ``usage.input_tokens``. Clef's ``output_tokens`` is always 0."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0
    raw = usage.get("input_tokens")
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _retry_after(headers: dict[str, str]) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _top_gap(probabilities: dict[str, float]) -> float:
    ordered = sorted(probabilities.values(), reverse=True)
    if len(ordered) < 2:
        return 1.0
    return ordered[0] - ordered[1]


def _verdict_to_cache(verdict: QualityVerdict) -> dict[str, Any]:
    return {
        "label": verdict.label,
        "probabilities": verdict.probabilities,
        "confidence": verdict.confidence,
        "confidence_raw": verdict.confidence_raw,
        "reasons": verdict.reasons,
        "abstained": verdict.abstained,
    }


def _verdict_from_cache(payload: dict[str, Any]) -> QualityVerdict:
    return QualityVerdict(
        label=str(payload["label"]),
        probabilities={str(key): float(value) for key, value in payload["probabilities"].items()},
        confidence=float(payload["confidence"]),
        reasons=[str(reason) for reason in payload.get("reasons", [])],
        abstained=bool(payload.get("abstained")),
        confidence_raw=_optional_float(payload.get("confidence_raw")),
    )


def _keeper_to_cache(verdict: KeeperVerdict) -> dict[str, Any]:
    return {
        "keeper_path": verdict.keeper_path,
        "probabilities": verdict.probabilities,
        "confidence": verdict.confidence,
        "abstained": verdict.abstained,
        "reasons": verdict.reasons,
        "needs_review": verdict.needs_review,
        "confidence_raw": verdict.confidence_raw,
    }


def _keeper_from_cache(payload: dict[str, Any]) -> KeeperVerdict:
    keeper = payload.get("keeper_path")
    return KeeperVerdict(
        keeper_path=None if keeper is None else str(keeper),
        probabilities={str(key): float(value) for key, value in payload["probabilities"].items()},
        confidence=float(payload["confidence"]),
        abstained=bool(payload["abstained"]),
        reasons=[str(reason) for reason in payload.get("reasons", [])],
        needs_review=bool(payload.get("needs_review")),
        confidence_raw=_optional_float(payload.get("confidence_raw")),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
