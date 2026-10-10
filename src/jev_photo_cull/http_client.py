"""Small POST helper so backends can be tested without a live socket."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

UrlOpen = Callable[..., Any]


def post_bytes(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
            header_map = {key.lower(): value for key, value in response.headers.items()}
            return status, header_map, response.read()
    except urllib.error.HTTPError as error:
        header_map = {key.lower(): value for key, value in error.headers.items()}
        return int(error.code), header_map, error.read()


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> tuple[int, dict[str, str], bytes]:
    data = json.dumps(payload).encode("utf-8")
    sent = dict(headers)
    sent.setdefault("Content-Type", "application/json")
    sent.setdefault("Accept", "application/json")
    return post_bytes(url, sent, data, timeout, urlopen=urlopen)
