from rizemind.web3.config import Web3Config
from rizemind.web3.indexer.block_watcher import BlockWatcher, LatestBlockBus
from web3 import AsyncWeb3


class Indexer:
    _block_watcher: BlockWatcher
    _block_bus: LatestBlockBus
    w3: AsyncWeb3

    def __init__(self, web3_config: Web3Config):
        self.w3 = web3_config.get_async_web3()
        self._block_bus = LatestBlockBus()
        self._block_watcher = BlockWatcher(self.w3, self._block_bus)

    def start(self):
        self._block_watcher.start()
