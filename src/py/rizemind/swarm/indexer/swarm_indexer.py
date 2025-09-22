import asyncio
import contextlib

from pyee.asyncio import AsyncIOEventEmitter
from rizemind.swarm.indexer.phases.latest_phase_bus import LatestPhaseBus
from rizemind.swarm.indexer.phases.new_phase_event import NewPhaseEvent
from rizemind.swarm.indexer.phases.phase_watcher import PhaseWatcher
from rizemind.swarm.swarm import Swarm
from rizemind.workflow.async_runtime import AsyncRuntime


class SwarmIndexer:
    swarm: Swarm
    ee: AsyncIOEventEmitter

    phase_watcher: PhaseWatcher
    bus: LatestPhaseBus

    def __init__(
        self,
        swarm: Swarm,
        runtime: AsyncRuntime,
        ee: AsyncIOEventEmitter,
    ):
        self.swarm = swarm
        self.ee = ee
        self._stop = asyncio.Event()
        self._started = False

    async def run(self) -> None:
        """Run until stop is requested. Must be called on the indexer loop thread."""
        if self._started:
            return
        self._started = True

        self.bus = LatestPhaseBus()
        self.phase_watcher = PhaseWatcher(self.swarm, self.bus)

        self.phase_watcher.start()

        try:
            # Wait until someone calls request_stop()
            await self._stop.wait()
        finally:
            # Graceful shutdown: cancel and await sub-tasks
            with contextlib.suppress(Exception):
                await self.phase_watcher.stop()
            self._started = False

    def request_stop(self) -> None:
        """May be called from any thread via loop.call_soon_threadsafe."""
        self._stop.set()
