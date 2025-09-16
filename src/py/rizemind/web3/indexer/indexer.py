from rizemind.web3.config import Web3Config
from rizemind.web3.indexer.block_watcher import BlockWatcher, LatestBlockBus
from rizemind.web3.indexer.logs.event_bus import EventBus
from rizemind.web3.indexer.logs.log_poller import LogPoller
from web3 import AsyncWeb3


class Indexer:
    _block_watcher: BlockWatcher
    _block_bus: LatestBlockBus
    _event_bus: EventBus
    _log_poller: LogPoller
    _w3: AsyncWeb3

    def __init__(self, web3_config: Web3Config):
        self._w3 = web3_config.get_async_web3()
        self._block_bus = LatestBlockBus()
        self._block_watcher = BlockWatcher(self._w3, self._block_bus)
        self._event_bus = EventBus(self._w3)
        self._log_poller = LogPoller(self._w3, self._block_bus, self._event_bus)

    async def start(self):
        self._block_watcher.start()
        self._log_poller.start()
