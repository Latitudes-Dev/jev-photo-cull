"""Clef client tests. No live Cloudflare credentials or network."""

import json
import urllib.error

import pytest

from jev_photo_cull.backends.clef import (
    ClefBackend,
    cloudflare_credentials,
    credentials_present,
    unwrap_workers_ai,
    workers_ai_url,
)
from jev_photo_cull.backends.local import LocalBackend, encode_multipart
from jev_photo_cull.cache import DecisionCache
from jev_photo_cull.decisions import build_quality_request, encode_image
from jev_photo_cull.errors import ConfigError, PhotoCullError
from jev_photo_cull.rate_limit import RateLimiter
from jev_photo_cull.records import ImageRecord
from tests.library import sharp_scene


def _jpeg_record(name: str, label_hint: str = "clean") -> ImageRecord:
    import io

    image = sharp_scene()
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    blob = buffer.getvalue()
    return ImageRecord(
        relative_path=name,
        absolute_path="/virtual/" + name,
        sha256=label_hint + name,
        file_size=len(blob),
        perceptual_hash=1,
        width=320,
        height=240,
        mean_luminance=128,
        luminance_std=40,
        shadow_fraction=0.1,
        highlight_fraction=0.1,
        rgb_mean=(128, 128, 128),
        sharpness=400,
        model_jpeg=blob,
        thumb_jpeg=blob,
    )


def _envelope(answers: dict) -> bytes:
    return json.dumps(
        {
            "success": True,
            "errors": [],
            "messages": [],
            "result": {
                "model": "clef-flash",
                "answers": answers,
                "usage": {"input_tokens": 32, "output_tokens": 0},
            },
        }
    ).encode()


def _quality_answer(label: str) -> dict:
    probabilities = {name: 0.02 for name in ("clean", "blurry", "too_dark", "overexposed", "unknown")}
    probabilities[label] = 0.92
    return {
        "type": "choice",
        "choice": label,
        "confidence": 0.92,
        "probabilities": probabilities,
    }


def test_workers_ai_url_matches_the_documented_path():
    assert workers_ai_url("abc123", "clef-flash") == (
        "https://api.cloudflare.com/client/v4/accounts/abc123/ai/run/@cf/cloudflare/clef-flash"
    )
    assert workers_ai_url("abc123", "clef").endswith("/ai/run/@cf/cloudflare/clef")
    with pytest.raises(ConfigError):
        workers_ai_url("acc/ount", "clef-flash")


def test_image_object_matches_clef_schema():
    encoded = encode_image(b"\xff\xd8\xff", "content-type")
    assert encoded == {"content_type": "image/jpeg", "base64": "/9j/"}
    record = _jpeg_record("one.jpg")
    body = build_quality_request("clef-flash", [record], image_format="content-type")
    assert body["model"] == "clef-flash"
    assert body["questions"]["quality_0"]["type"] == "choice"
    assert set(body["questions"]["quality_0"]["criteria"]) == {
        "clean",
        "blurry",
        "too_dark",
        "overexposed",
        "unknown",
    }
    assert body["images"][0]["content_type"] == "image/jpeg"
    assert "state" in body and "questions" in body


def test_missing_credentials_do_not_call_the_network(tmp_path):
    calls = []

    def post(*args):
        calls.append(args)
        raise AssertionError("network")

    with pytest.raises(ConfigError) as caught:
        ClefBackend(cache=DecisionCache(tmp_path / "c.json"), environ={}, post=post)
    message = str(caught.value)
    assert "CLOUDFLARE_ACCOUNT_ID" in message
    assert "CLOUDFLARE_API_TOKEN" in message
    assert "CLOUDFLARE_AUTH_TOKEN" in message
    assert calls == []


def test_auth_token_env_is_accepted_as_documented_fallback():
    account, token = cloudflare_credentials(
        {"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_AUTH_TOKEN": "from-docs"}
    )
    assert (account, token) == ("acc", "from-docs")
    assert credentials_present({"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "primary"})
    assert not credentials_present({})


def test_clef_request_shape_batching_and_cache(tmp_path):
    records = [_jpeg_record(f"p{index}.jpg", label_hint=str(index)) for index in range(5)]
    seen = []

    def post(url, headers, payload, timeout):
        seen.append((url, headers, payload, timeout))
        answers = {
            key: _quality_answer("blurry" if "blurry" in question["instructions"] else "clean")
            for key, question in payload["questions"].items()
        }
        if "keeper" in payload["questions"]:
            criteria = payload["questions"]["keeper"]["criteria"]
            answers["keeper"] = {
                "type": "choice",
                "choice": "shot_1",
                "confidence": 0.88,
                "probabilities": {option: (0.88 if option == "shot_1" else 0.03) for option in criteria},
            }
        return 200, {}, _envelope(answers)

    backend = ClefBackend(
        cache=DecisionCache(tmp_path / "cache.json"),
        requests_per_minute=0,
        environ={"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "secret-token"},
        post=post,
    )
    verdicts = backend.classify(records)
    assert len(seen) == 2
    assert [len(call[2]["images"]) for call in seen] == [4, 1]
    url, headers, payload, _timeout = seen[0]
    assert url == "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash"
    assert headers["Authorization"] == "Bearer secret-token"
    assert payload["model"] == "clef-flash"
    assert payload["images"][0]["content_type"] == "image/jpeg"
    import base64

    assert base64.standard_b64decode(payload["images"][0]["base64"]).startswith(b"\xff\xd8")
    assert verdicts["p0.jpg"].label == "clean"
    assert "secret-token" not in json.dumps(payload["state"])

    from jev_photo_cull.records import Group

    group = Group(id="g0001", members=records[:2], kind="near-duplicate")
    keeper = backend.choose_keepers([group])["g0001"]
    assert keeper.keeper_path == records[0].relative_path
    assert not keeper.abstained

    before = len(seen)
    again = backend.classify(records)
    assert len(seen) == before
    assert again["p0.jpg"].label == "clean"


def test_api_error_envelope_is_reported(tmp_path):
    def fail(status, message):
        return lambda *args: (
            status,
            {},
            json.dumps({"success": False, "errors": [{"code": 5007, "message": message}], "result": None}).encode(),
        )

    record = _jpeg_record("a.jpg")
    backend = ClefBackend(
        cache=DecisionCache(tmp_path / "cache.json"),
        requests_per_minute=0,
        environ={"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "t"},
        post=fail(200, "No such model"),
        sleep=lambda _seconds: None,
    )
    with pytest.raises(PhotoCullError, match="No such model"):
        backend.classify([record])
    assert not (tmp_path / "cache.json").exists()


def test_retries_retryable_status_then_succeeds(tmp_path):
    record = _jpeg_record("a.jpg")
    attempts = {"n": 0}
    slept = []

    def post(url, headers, payload, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return 503, {"retry-after": "0"}, b'{"success": false, "errors": [{"message": "busy"}]}'
        return 200, {}, _envelope({"quality_0": _quality_answer("too_dark")})

    backend = ClefBackend(
        cache=DecisionCache(tmp_path / "c.json"),
        requests_per_minute=0,
        environ={"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "t"},
        post=post,
        sleep=slept.append,
    )
    verdict = backend.classify([record])["a.jpg"]
    assert verdict.label == "too_dark"
    assert attempts["n"] == 3
    assert slept == [0.0, 0.0]


def test_unwrap_accepts_bare_answers_and_rejects_garbage():
    bare = json.dumps(
        {"model": "clef-flash", "answers": {"quality_0": _quality_answer("clean")}, "usage": {"input_tokens": 1, "output_tokens": 0}}
    ).encode()
    parsed = unwrap_workers_ai(200, {}, bare)
    assert parsed["answers"]["quality_0"]["choice"] == "clean"
    with pytest.raises(PhotoCullError):
        unwrap_workers_ai(200, {}, b'{"success": true, "result": {"response": "hello"}}')


def test_rate_limiter_waits_between_calls():
    now = {"t": 0.0}
    slept = []

    def clock():
        return now["t"]

    def sleep(seconds):
        slept.append(seconds)
        now["t"] += seconds

    limiter = RateLimiter(30, sleep=sleep, clock=clock)
    limiter.wait()
    limiter.wait()
    assert slept == [2.0]
    assert now["t"] == 2.0


def test_local_systemone_sends_raw_base64_and_explains_connection_errors(tmp_path):
    record = _jpeg_record("frame.jpg")
    seen = {}

    def post(url, headers, payload, timeout):
        seen["url"] = url
        seen["payload"] = payload
        return 200, {}, json.dumps(
            {"model": "clef-flash", "answers": {"quality_0": _quality_answer("clean")}, "usage": {"input_tokens": 1, "output_tokens": 0}}
        ).encode()

    backend = LocalBackend(
        cache=DecisionCache(tmp_path / "c.json"),
        url="http://127.0.0.1:11434/v1/systemone",
        model="clef-flash",
        protocol="systemone",
        image_format="raw",
        requests_per_minute=0,
        post=post,
    )
    assert backend.classify([record])["frame.jpg"].label == "clean"
    assert seen["url"].endswith("/v1/systemone")
    assert isinstance(seen["payload"]["images"][0], str)
    assert not str(seen["payload"]["images"][0]).startswith("data:")

    def down(*args):
        raise urllib.error.URLError("connection refused")

    offline = LocalBackend(
        cache=DecisionCache(tmp_path / "c2.json"),
        protocol="systemone",
        requests_per_minute=0,
        post=down,
    )
    with pytest.raises(PhotoCullError, match="ollama pull clef-flash"):
        offline.classify([record])


def test_imajev_protocol_is_multipart_and_one_image(tmp_path):
    record = _jpeg_record("frame.jpg")
    seen = {}

    def multipart(url, headers, body, timeout):
        seen["body"] = body
        seen["type"] = headers["Content-Type"]
        answers = {"quality_0": _quality_answer("blurry")}
        answers["quality_0"]["abstained"] = False
        return 200, {}, json.dumps({"model": "imajev-4b", "answers": answers}).encode()

    backend = LocalBackend(
        cache=DecisionCache(tmp_path / "c.json"),
        protocol="imajev",
        model="imajev-4b",
        requests_per_minute=0,
        post_multipart=multipart,
    )
    verdict = backend.classify([record])["frame.jpg"]
    assert verdict.label == "blurry"
    assert seen["type"].startswith("multipart/form-data")
    assert b'name="request"' in seen["body"]
    assert b'name="image"' in seen["body"]
    assert b"quality_0" in seen["body"]
    body, content_type = encode_multipart('{"ok":true}', "photo.jpg", b"\xff\xd8")
    assert content_type.startswith("multipart/form-data")
    assert b'{"ok":true}' in body
