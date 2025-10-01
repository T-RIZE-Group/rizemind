import asyncio
import contextlib
import random

from rizemind.web3.indexer.blocks.latest_block_bus import LatestBlockBus
from web3 import AsyncWeb3

CONFIRMATIONS = 6
MIN_POLL = 0.5  # seconds
MAX_POLL = 4.0  # seconds


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
            except asyncio.CancelledError:
                # Re-raise cancellation to allow proper task shutdown
                raise
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
