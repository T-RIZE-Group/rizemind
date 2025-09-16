import asyncio
import contextlib
import random
from collections.abc import AsyncIterator

from web3 import AsyncWeb3

CONFIRMATIONS = 6
MIN_POLL = 0.5  # seconds
MAX_POLL = 4.0  # seconds


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


class BlockWatcher:
    """Polls eth_blockNumber and publishes the latest when it increases."""

    def __init__(self, w3: AsyncWeb3, bus: LatestBlockBus):
        self.w3 = w3
        self.bus = bus
        self._task: asyncio.Task | None = None

    async def _poll_loop(self):
        interval = MIN_POLL
        last_seen: int | None = None
        while True:
            try:
                latest = await self.w3.eth.block_number
                if last_seen is None or latest > last_seen:
                    await self.bus.publish(latest)
                    last_seen = latest
                # adaptive sleep: faster while catching up, slower once steady
                interval = (
                    MIN_POLL
                    if last_seen is None or latest == last_seen
                    else min(MAX_POLL, interval * 1.2)
                )
            except Exception:
                # transient RPC errors → back off with jitter
                interval = min(MAX_POLL, max(MIN_POLL, interval * 1.5))
            await asyncio.sleep(interval + random.uniform(0, 0.2))

    def start(self):
        self._task = asyncio.create_task(self._poll_loop(), name="BlockWatcher")

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
