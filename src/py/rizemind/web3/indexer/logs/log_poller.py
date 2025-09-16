import asyncio
import contextlib

from rizemind.web3.indexer.block_watcher import LatestBlockBus
from rizemind.web3.indexer.logs.event_bus import EventBus
from web3 import AsyncWeb3
from web3.types import FilterParams

LOG_BATCH = 1000


class LogPoller:
    """
    Processes logs strictly in order from (last+1 .. latest), with no rollback logic.
    Checkpoints only the last processed block number.
    """

    _task: asyncio.Task | None = None
    w3: AsyncWeb3
    bus: LatestBlockBus
    event_bus: EventBus
    addresses: list[str] | None
    topics: list[str] | None
    batch_size: int
    block_height: int

    def __init__(
        self,
        w3: AsyncWeb3,
        bus: LatestBlockBus,
        event_bus: EventBus,
        addresses: list[str] | None = None,
        topics: list[str] | None = None,
        batch_size: int = LOG_BATCH,
        start_block: int = -1,  # bootstrap if no checkpoint found
    ):
        self.w3 = w3
        self.bus = bus
        self.event_bus = event_bus
        self.batch_size = batch_size
        self.block_height = start_block
        self._task = None

    async def _handle_range(self, from_block: int, to_block: int):
        start = max(0, from_block)
        while start <= to_block:
            end = min(start + self.batch_size - 1, to_block)
            params = FilterParams({"fromBlock": start, "toBlock": end})

            logs = await self.w3.eth.get_logs(params)
            for log in logs:
                topics = log.get("topics", [])
                if not topics:
                    continue
                await self.event_bus.publish_raw_log(log)

            start = end + 1

    async def _run(self):
        version = 0
        async for version, latest in self.bus.stream(version):
            if latest > self.block_height:
                await self._handle_range(self.block_height + 1, latest)
                self.block_height = latest

    def start(self):
        self._task = asyncio.create_task(self._run(), name="LogPollerConsumer")

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
