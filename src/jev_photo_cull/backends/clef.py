"""Cloudflare Workers AI Clef backend.

Documented endpoint (Clef model page, fetched 2026-10-09):

    POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef-flash
    Authorization: Bearer $TOKEN
    Content-Type: application/json

The JSON body is the model input itself (no ``input`` wrapper): ``model``,
``state``, ``questions``, and optional ``images``. ``model`` is ``clef`` or
``clef-flash`` and must match the model id in the URL.

Successful Workers AI REST calls use the account API envelope
``{"success": true, "errors": [], "messages": [], "result": {...}}``.
The Clef result object is ``model``, ``answers``, and ``usage``
(schema-output.json). A bare body that already has ``answers`` is accepted
so a binding-style payload can be parsed in tests.

Image limits from the same schema: at most 4 PNG/JPEG/WebP images, 4 MiB and
16 megapixels each, 8 MiB decoded total, whole body at most 13 MiB.
This client sends downscaled JPEGs as ``{"content_type": "image/jpeg", "base64": "..."}``.
"""

from __future__ import annotations

import json
import os
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jev_photo_cull.backends.remote import DecisionClient, PostJson
from jev_photo_cull.cache import DecisionCache
from jev_photo_cull.errors import ConfigError, PhotoCullError
from jev_photo_cull.http_client import post_json
from jev_photo_cull.rate_limit import RateLimiter

API_ROOT = "https://api.cloudflare.com/client/v4/accounts"
MODEL_IDS = {
    "clef-flash": "@cf/cloudflare/clef-flash",
    "clef": "@cf/cloudflare/clef",
}


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from ``.env`` without overriding the real environment."""
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def cloudflare_credentials(environ: dict[str, str] | None = None) -> tuple[str, str]:
    env = os.environ if environ is None else environ
    account = env.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = env.get("CLOUDFLARE_API_TOKEN", "").strip() or env.get("CLOUDFLARE_AUTH_TOKEN", "").strip()
    return account, token


def credentials_present(environ: dict[str, str] | None = None) -> bool:
    account, token = cloudflare_credentials(environ)
    return bool(account and token)


def require_credentials(environ: dict[str, str] | None = None) -> tuple[str, str]:
    account, token = cloudflare_credentials(environ)
    if not account or not token:
        raise ConfigError(
            "Clef needs Cloudflare credentials. Set CLOUDFLARE_ACCOUNT_ID and "
            "CLOUDFLARE_API_TOKEN. CLOUDFLARE_AUTH_TOKEN is also accepted "
            "(that is the name in the Workers AI Clef examples) when "
            "CLOUDFLARE_API_TOKEN is unset. No request was sent."
        )
    return account, token


def workers_ai_url(account_id: str, model_name: str) -> str:
    if model_name not in MODEL_IDS:
        raise ConfigError(f"unknown Clef model {model_name!r}. Use clef-flash or clef.")
    if not account_id or any(char in account_id for char in "/\\? #\t"):
        raise ConfigError("CLOUDFLARE_ACCOUNT_ID is missing or not a single account id.")
    # Slashes in the model id are part of the documented path, not extra segments to escape.
    return f"{API_ROOT}/{account_id}/ai/run/{MODEL_IDS[model_name]}"


def unwrap_workers_ai(status: int, _headers: dict[str, str], raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text) if text else None
    except json.JSONDecodeError as exc:
        raise PhotoCullError(f"Workers AI returned HTTP {status} with a non-JSON body") from exc
    if not isinstance(payload, dict):
        raise PhotoCullError(f"Workers AI returned HTTP {status} without a JSON object")
    if "success" in payload or "errors" in payload:
        if not payload.get("success", False):
            errors = payload.get("errors") or []
            message = "; ".join(
                str(item.get("message", item)) if isinstance(item, dict) else str(item) for item in errors
            )
            raise PhotoCullError(
                f"Workers AI error (HTTP {status}): {message or 'request failed'}"
            )
        result = payload.get("result")
        if isinstance(result, dict) and "answers" in result:
            return result
        raise PhotoCullError(
            "Workers AI succeeded but the result had no Clef answers object. "
            f"Result keys: {sorted(result) if isinstance(result, dict) else type(result).__name__}."
        )
    if "answers" in payload:
        return payload
    raise PhotoCullError(
        f"Unrecognized Workers AI response (HTTP {status}). Keys: {sorted(payload)}."
    )


class ClefBackend:
    """Decision backend. Constructing it without credentials raises ConfigError."""

    name = "clef"

    def __init__(
        self,
        *,
        model: str = "clef-flash",
        cache: DecisionCache,
        requests_per_minute: float = 30.0,
        timeout: float = 120.0,
        review_below: float = 0.60,
        environ: dict[str, str] | None = None,
        post: PostJson | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if model not in MODEL_IDS:
            raise ConfigError(f"unknown Clef model {model!r}. Use clef-flash or clef.")
        account, token = require_credentials(environ)
        self.model_id = MODEL_IDS[model]
        self.model = model
        limiter_kwargs: dict[str, Any] = {}
        if sleep is not None:
            limiter_kwargs["sleep"] = sleep
        if clock is not None:
            limiter_kwargs["clock"] = clock
        self._client = DecisionClient(
            name="clef",
            model=model,
            image_format="content-type",
            cache=cache,
            limiter=RateLimiter(requests_per_minute, **limiter_kwargs),
            post=post or _default_post,
            url=workers_ai_url(account, model),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            unwrap=unwrap_workers_ai,
            review_below=review_below,
            sleep=sleep or __import__("time").sleep,
        )

    @property
    def calls(self) -> int:
        return self._client.calls

    @property
    def usage(self) -> dict[str, Any]:
        return self._client.usage

    def classify(self, records: list) -> dict:
        return self._client.classify(records)

    def choose_keepers(self, groups: list) -> dict:
        return self._client.choose_keepers(groups)


def _default_post(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    try:
        return post_json(url, headers, payload, timeout)
    except urllib.error.URLError as exc:
        raise PhotoCullError(f"Could not reach Workers AI: {exc.reason}") from exc
