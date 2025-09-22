import asyncio
from collections.abc import AsyncIterator


class LatestBlockBus:
    """Fan-out latest block number; drops intermediates, never grows a queue."""

    def __init__(self):
        self._cond = asyncio.Condition()
        self._version = 0
        self._latest_block: int | None = None

    async def publish(self, latest_block: int):
        async with self._cond:
            if self._latest_block is None or latest_block > self._latest_block:
                self._latest_block = latest_block
                self._version += 1
                self._cond.notify_all()

    async def stream(self, since_version: int = 0) -> AsyncIterator[tuple[int, int]]:
        while True:
            async with self._cond:
                while self._version == since_version:
                    await self._cond.wait()
                since_version = self._version
                assert self._latest_block is not None
                lb = self._latest_block
            yield since_version, lb  # yield outside the lock
