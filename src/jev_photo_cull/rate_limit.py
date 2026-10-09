"""Client-side gap between decision requests. Workers AI account limits are separate."""

from __future__ import annotations

import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: float,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_minute < 0:
            raise ValueError("requests_per_minute must be >= 0")
        self.interval = 0.0 if requests_per_minute == 0 else 60.0 / requests_per_minute
        self._sleep = sleep
        self._clock = clock
        self._next = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._clock()
        self._next = now + self.interval
