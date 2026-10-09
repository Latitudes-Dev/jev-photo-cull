"""JSON cache of decision results, keyed by file hash and request schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DecisionCache:
    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._data: dict[str, Any] = {}
        if enabled and path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._data = loaded

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
