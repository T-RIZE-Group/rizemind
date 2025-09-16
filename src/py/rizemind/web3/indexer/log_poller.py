import asyncio, random, contextlib
from typing import AsyncIterator, Optional
from rizemind.web3.indexer.block_watcher import LatestBlockBus
from web3 import AsyncWeb3
from web3.providers.rpc import AsyncHTTPProvider


class LogPollerConsumer:
    """
    Processes logs strictly in order from (last+1 .. latest), with no rollback logic.
    Checkpoints only the last processed block number.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        bus: LatestBlockBus,
        checkpoint_path: str = "logs.ckpt",
        addresses: list[str] | None = None,
        topics: list[str] | None = None,
        batch_size: int = LOG_BATCH,
        start_block: int = -1,  # bootstrap if no checkpoint found
    ):
        self.w3 = w3
        self.bus = bus
        self.ckpt_path = checkpoint_path
        self.addresses = addresses
        self.topics = topics
        self.batch_size = batch_size
        self.start_block = start_block
        self._task: Optional[asyncio.Task] = None

    async def _load_ckpt(self) -> int:
        try:
            with open(self.ckpt_path) as f:
                return int(f.read().strip())
        except FileNotFoundError:
            return self.start_block

    async def _save_ckpt(self, n: int):
        with open(self.ckpt_path, "w") as f:
            f.write(str(n))

    async def _handle_range(self, from_block: int, to_block: int):
        b = max(0, from_block)
        while b <= to_block:
            e = min(b + self.batch_size - 1, to_block)
            params = {"fromBlock": b, "toBlock": e}
            if self.addresses:
                params["address"] = self.addresses
            if self.topics:
                params["topics"] = [self.topics]  # OR at position 0
            logs = await self.w3.eth.get_logs(params)
            # TODO: route to subscribers (decode & dispatch)
            # for lg in logs: route(lg)
            b = e + 1

    async def _run(self):
        last = await self._load_ckpt()
        version = 0
        async for version, latest in self.bus.stream(version):
            if latest > last:
                await self._handle_range(last + 1, latest)
                last = latest
                await self._save_ckpt(last)

    def start(self):
        self._task = asyncio.create_task(self._run(), name="LogPollerConsumer")

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
