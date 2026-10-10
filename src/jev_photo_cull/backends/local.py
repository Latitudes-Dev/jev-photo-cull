"""Optional local decision server.

Default protocol is Ollama's System One endpoint (``POST /v1/systemone``) with
raw base64 images, which is what the Ollama clef-flash card documents.
``--local-protocol imajev`` matches the imajev README: one JPEG per multipart
request (``image`` file plus a ``request`` JSON field) against a local
imajev server. imajev's README shows a single image field and at most two
photos; keeper choice for that protocol stays on the local heuristics.

Nothing here downloads weights. If the server is not running, the error says
how to start one.
"""

from __future__ import annotations

import json
import os
import urllib.error
import uuid
from collections.abc import Callable
from typing import Any

from jev_photo_cull.backends.heuristics_backend import HeuristicsBackend
from jev_photo_cull.backends.remote import (
    DecisionClient,
    PostJson,
    input_tokens_from_payload,
    neurons_from_headers,
)
from jev_photo_cull.cache import DecisionCache
from jev_photo_cull.decisions import (
    DECISION_SCORE_VERSION,
    QUALITY_CRITERIA,
    build_quality_request,
    parse_choice,
)
from jev_photo_cull.errors import ConfigError, PhotoCullError
from jev_photo_cull.http_client import post_bytes
from jev_photo_cull.rate_limit import RateLimiter
from jev_photo_cull.records import ImageRecord
from jev_photo_cull.verdicts import QualityVerdict

DEFAULT_URL = "http://127.0.0.1:11434/v1/systemone"
DEFAULT_MODEL = "clef-flash"


def local_settings(environ: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    return {
        "url": env.get("JEV_PHOTO_CULL_LOCAL_URL", DEFAULT_URL).strip() or DEFAULT_URL,
        "model": env.get("JEV_PHOTO_CULL_LOCAL_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "protocol": env.get("JEV_PHOTO_CULL_LOCAL_PROTOCOL", "systemone").strip() or "systemone",
        "image_format": env.get("JEV_PHOTO_CULL_LOCAL_IMAGE_FORMAT", "raw").strip() or "raw",
    }


def encode_multipart(request_json: str, filename: str, jpeg: bytes) -> tuple[bytes, str]:
    boundary = "jevphotocull" + uuid.uuid4().hex
    parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="request"\r\n\r\n'
            f"{request_json}\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8")
        + jpeg
        + b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def unwrap_local(status: int, _headers: dict[str, str], raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="replace")
    if status >= 400:
        raise PhotoCullError(f"local decision server returned HTTP {status}: {text[:500]}")
    try:
        payload = json.loads(text) if text else None
    except json.JSONDecodeError as exc:
        raise PhotoCullError("local decision server returned a non-JSON body") from exc
    if isinstance(payload, dict) and "answers" in payload:
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        result = payload["result"]
        if "answers" in result:
            return result
    raise PhotoCullError("local decision response did not include answers")


class LocalBackend:
    name = "local"

    def __init__(
        self,
        *,
        cache: DecisionCache,
        url: str | None = None,
        model: str | None = None,
        protocol: str | None = None,
        image_format: str | None = None,
        requests_per_minute: float = 60.0,
        timeout: float = 180.0,
        review_below: float = 0.60,
        environ: dict[str, str] | None = None,
        post: PostJson | None = None,
        post_multipart: Callable[..., tuple[int, dict[str, str], bytes]] | None = None,
        sleep: Callable[[float], None] | None = None,
        heuristics: HeuristicsBackend | None = None,
    ) -> None:
        settings = local_settings(environ)
        self.url = url or settings["url"]
        self.model_id = model or settings["model"]
        self.protocol = (protocol or settings["protocol"]).lower()
        self.image_format = image_format or settings["image_format"]
        if self.protocol not in {"systemone", "imajev"}:
            raise ConfigError("local protocol must be systemone or imajev")
        if self.image_format not in {"raw", "data-url", "content-type"}:
            raise ConfigError("local image format must be raw, data-url, or content-type")
        self._heuristics = heuristics or HeuristicsBackend(review_below=review_below)
        self._multipart = post_multipart or _default_multipart
        self.cache = cache
        self.timeout = timeout
        self.review_below = review_below
        if self.protocol == "systemone":
            self._client: DecisionClient | None = DecisionClient(
                name="local",
                model=self.model_id,
                image_format=self.image_format,
                cache=cache,
                limiter=RateLimiter(requests_per_minute, **({"sleep": sleep} if sleep else {})),
                post=post or _default_post,
                url=self.url,
                headers={},
                timeout=timeout,
                unwrap=unwrap_local,
                review_below=review_below,
                sleep=sleep or __import__("time").sleep,
            )
        else:
            self._client = None
            self._post = post or _default_post
            self._limiter = RateLimiter(requests_per_minute, **({"sleep": sleep} if sleep else {}))
            self._imajev_calls = 0
            self._imajev_neurons = 0.0
            self._imajev_input_tokens = 0

    @property
    def usage(self) -> dict[str, Any]:
        if self._client is not None:
            return self._client.usage
        return {
            "calls": self._imajev_calls,
            "neurons": round(self._imajev_neurons, 4),
            "input_tokens": self._imajev_input_tokens,
        }

    @property
    def calls(self) -> int:
        if self._client is not None:
            return self._client.calls
        return self._imajev_calls

    def classify(self, records: list[ImageRecord]) -> dict[str, QualityVerdict]:
        if self._client is not None:
            try:
                return self._client.classify(records)
            except urllib.error.URLError as exc:
                raise self._unreachable(exc) from exc
        return self._classify_imajev(records)

    def choose_keepers(self, groups: list):
        if self.protocol == "imajev":
            # The published imajev request uses one image file. Keeper choice
            # stays on local measurements rather than an undocumented multi-image field.
            return self._heuristics.choose_keepers(groups)
        assert self._client is not None
        try:
            return self._client.choose_keepers(groups)
        except urllib.error.URLError as exc:
            raise self._unreachable(exc) from exc

    def _classify_imajev(self, records: list[ImageRecord]) -> dict[str, QualityVerdict]:
        verdicts: dict[str, QualityVerdict] = {}
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
            key = f"quality|imajev|{DECISION_SCORE_VERSION}|{self.model_id}|{record.sha256}"
            cached = self.cache.get(key)
            if isinstance(cached, dict):
                raw_confidence = cached.get("confidence_raw")
                verdicts[record.relative_path] = QualityVerdict(
                    label=str(cached["label"]),
                    probabilities={str(k): float(v) for k, v in cached["probabilities"].items()},
                    confidence=float(cached["confidence"]),
                    reasons=[str(reason) for reason in cached.get("reasons", [])],
                    abstained=bool(cached.get("abstained")),
                    confidence_raw=None if raw_confidence is None else float(raw_confidence),
                )
                continue
            request = build_quality_request(self.model_id, [record], image_format="raw")
            # imajev takes the image as a file part, not an images array.
            request.pop("images", None)
            body, content_type = encode_multipart(
                json.dumps(request),
                "photo.jpg",
                record.model_jpeg,
            )
            self._limiter.wait()
            self._imajev_calls += 1
            try:
                status, headers, raw = self._multipart(
                    self.url,
                    {"Content-Type": content_type, "Accept": "application/json"},
                    body,
                    self.timeout,
                )
            except urllib.error.URLError as exc:
                raise self._unreachable(exc) from exc
            self._imajev_neurons += neurons_from_headers(headers)
            payload = unwrap_local(status, headers, raw)
            self._imajev_input_tokens += input_tokens_from_payload(payload)
            parsed = parse_choice(
                payload["answers"].get("quality_0"),
                expected_options=set(QUALITY_CRITERIA),
            )
            label = "unknown" if parsed.abstained else parsed.choice
            verdict = QualityVerdict(
                label=label,
                probabilities=parsed.probabilities,
                confidence=parsed.score,
                reasons=[f"imajev {self.model_id} chose {label} (p={parsed.score:.2f})"],
                abstained=parsed.abstained,
                confidence_raw=parsed.confidence_raw,
            )
            verdicts[record.relative_path] = verdict
            self.cache.put(
                key,
                {
                    "label": verdict.label,
                    "probabilities": verdict.probabilities,
                    "confidence": verdict.confidence,
                    "confidence_raw": verdict.confidence_raw,
                    "reasons": verdict.reasons,
                    "abstained": verdict.abstained,
                },
            )
        return verdicts

    def _unreachable(self, exc: urllib.error.URLError) -> PhotoCullError:
        return PhotoCullError(
            "Could not reach the local decision model at "
            f"{self.url} ({exc.reason}). The local backend is optional and this "
            "package does not download weights. For Ollama, run `ollama pull clef-flash` "
            "and POST to /v1/systemone (raw base64 images). For imajev, start its "
            "server and pass --local-protocol imajev."
        )


def _default_post(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    from jev_photo_cull.http_client import post_json

    return post_json(url, headers, payload, timeout)


def _default_multipart(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    try:
        return post_bytes(url, headers, body, timeout)
    except urllib.error.URLError as exc:
        raise PhotoCullError(
            f"Could not reach the local decision model at {url} ({exc.reason})."
        ) from exc
