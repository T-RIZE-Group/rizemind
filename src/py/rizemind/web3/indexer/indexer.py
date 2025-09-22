import asyncio
import contextlib
from typing import Self

from rizemind.web3.config import Web3Config
from rizemind.web3.indexer.blocks.block_watcher import BlockWatcher
from rizemind.web3.indexer.blocks.latest_block_bus import LatestBlockBus
from rizemind.web3.indexer.logs.event_bus import EventBus
from rizemind.web3.indexer.logs.log_poller import LogPoller
from web3 import AsyncWeb3


class Indexer:
    _block_watcher: BlockWatcher
    _block_bus: LatestBlockBus
    _event_bus: EventBus
    _log_poller: LogPoller
    _w3: AsyncWeb3 | None
    _w3_config: Web3Config

    def __init__(self, web3_config: Web3Config):
        self._block_bus = LatestBlockBus()
        self._w3_config = web3_config

        self._stop = asyncio.Event()
        self._started = False

    def get_block_bus(self) -> LatestBlockBus:
        return self._block_bus

    def get_event_bus(self) -> EventBus:
        return self._event_bus

    @classmethod
    def instance(cls, web3_config: Web3Config) -> Self:
        if not hasattr(cls, "_instance"):
            cls._instance = cls(web3_config)
        return cls._instance

    async def run(self) -> None:
        """Run until stop is requested. Must be called on the indexer loop thread."""
        if self._started:
            return
        self._started = True
        self._w3 = await self._w3_config.get_async_web3()
        self._block_watcher = BlockWatcher(self._w3, self._block_bus)
        self._event_bus = EventBus(self._w3)
        self._log_poller = LogPoller(self._w3, self._block_bus, self._event_bus)
        # Start sub-tasks ON THIS LOOP
        self._block_watcher.start()
        self._log_poller.start()

        try:
            # Wait until someone calls request_stop()
            await self._stop.wait()
        finally:
            # Graceful shutdown: cancel and await sub-tasks
            with contextlib.suppress(Exception):
                await self._log_poller.stop()
            with contextlib.suppress(Exception):
                await self._block_watcher.stop()
            self._started = False

    def request_stop(self) -> None:
        """May be called from any thread via loop.call_soon_threadsafe."""
        self._stop.set()
