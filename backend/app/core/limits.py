"""In-process limits for expensive operations (single-process deployments; use a shared store when scaling out)."""

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator

from fastapi import Request

from app.core.config import get_settings
from app.core.errors import TooManyRequestsError


class SlidingWindowLimiter:
    def __init__(self, max_calls: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.max_calls, self.window_s, self._clock = max_calls, window_s, clock
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            calls = self._calls[key]
            while calls and now - calls[0] >= self.window_s:
                calls.popleft()
            if len(calls) >= self.max_calls:
                retry = max(1, int(self.window_s - (now - calls[0])) + 1)
                raise TooManyRequestsError(
                    f"Too many route planning requests; limit is {self.max_calls} per {self.window_s:g} s",
                    headers={"Retry-After": str(retry)},
                )
            calls.append(now)

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


class ConcurrencyGate:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._semaphore = threading.BoundedSemaphore(limit)

    def acquire(self) -> None:
        if not self._semaphore.acquire(blocking=False):
            raise TooManyRequestsError(
                "The route planner is busy; retry shortly", headers={"Retry-After": "2"}
            )

    def release(self) -> None:
        self._semaphore.release()


_settings = get_settings()
route_rate_limiter = SlidingWindowLimiter(_settings.route_rate_limit_per_min)
route_gate = ConcurrencyGate(_settings.route_max_concurrent)


def route_planning_guard(request: Request) -> Iterator[None]:
    """FastAPI dependency: per-client rate limit plus a cap on simultaneous planner runs."""
    route_rate_limiter.check(request.client.host if request.client else "unknown")
    route_gate.acquire()
    try:
        yield
    finally:
        route_gate.release()
