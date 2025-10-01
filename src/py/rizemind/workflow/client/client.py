from collections.abc import Callable

from pyee.asyncio import AsyncIOEventEmitter
from rizemind.swarm.indexer.phases.new_phase_event import NewPhaseEvent
from rizemind.swarm.indexer.swarm_indexer import SwarmIndexer
from rizemind.swarm.indexer.swarm_indexer_runtime import SwarmIndexerRuntime
from rizemind.swarm.swarm import Swarm
from rizemind.web3.config import Web3Config
from rizemind.web3.indexer.indexer import Indexer
from rizemind.web3.indexer.indexer_runtime import IndexerRuntime
from rizemind.workflow.async_runtime import AsyncRuntime


class RizemindClient:
    async_runtime: AsyncRuntime
    web3_config: Web3Config
    swarm: Swarm
    ee = AsyncIOEventEmitter()

    def __init__(self, swarm: Swarm, web3_config: Web3Config):
        self.web3_config = web3_config
        self.swarm = swarm
        self.async_runtime = AsyncRuntime(name="RizemindClient")
        self.indexer = Indexer(self.web3_config)
        self.indexer_runtime = IndexerRuntime(self.indexer, self.async_runtime)
        self.swarm_indexer = SwarmIndexer(self.swarm, self.async_runtime, self.ee)
        self.swarm_indexer_runtime = SwarmIndexerRuntime(
            self.swarm_indexer, self.async_runtime
        )

    def register_phase_handler(
        self, event_name: str, handler: Callable[[NewPhaseEvent], None]
    ):
        self.ee.on(event_name, handler)

    def start(self):
        self.async_runtime.start()
        self.indexer_runtime.start()
        self.swarm_indexer_runtime.start()

    def stop(self):
        self.indexer_runtime.stop()
        self.swarm_indexer_runtime.stop()
        self.async_runtime.stop()
