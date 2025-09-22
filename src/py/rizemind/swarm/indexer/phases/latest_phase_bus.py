import asyncio
from collections.abc import AsyncIterator


class LatestPhaseBus:
    """Fan-out latest phase changes; drops intermediates, never grows a queue."""

    def __init__(self):
        self._cond = asyncio.Condition()
        self._version = 0
        self._latest_phase: str | None = None

    async def publish(self, latest_phase: str):
        async with self._cond:
            if self._latest_phase is None or latest_phase != self._latest_phase:
                self._latest_phase = latest_phase
                self._version += 1
                self._cond.notify_all()

    async def stream(self, since_version: int = 0) -> AsyncIterator[tuple[int, str]]:
        while True:
            async with self._cond:
                while self._version == since_version:
                    await self._cond.wait()
                since_version = self._version
                assert self._latest_phase is not None
                lp = self._latest_phase
            yield since_version, lp  # yield outside the lock
