import asyncio
import contextlib
import random

from rizemind.swarm.swarm import Swarm

from .latest_phase_bus import LatestPhaseBus

MIN_POLL = 1.0  # seconds - phases change slower than blocks
MAX_POLL = 10.0  # seconds
JITTER_MAX = 0.5  # seconds


class PhaseWatcher:
    """Polls swarm.get_current_phase() and publishes the latest when it changes."""

    def __init__(self, swarm: Swarm, bus: LatestPhaseBus):
        self.swarm = swarm
        self.bus = bus
        self._task: asyncio.Task | None = None

    async def _poll_loop(self):
        interval = MIN_POLL
        last_seen: str | None = None
        while True:
            try:
                current_phase = self.swarm.get_current_phase()
                if last_seen is None or current_phase != last_seen:
                    await self.bus.publish(current_phase)
                    last_seen = current_phase
                    # Reset interval on phase change for faster detection of subsequent changes
                    interval = MIN_POLL
                else:
                    # Gradually increase interval when no changes detected
                    interval = min(MAX_POLL, interval * 1.1)
            except asyncio.CancelledError:
                raise
            except Exception:
                # transient RPC errors → back off with jitter
                interval = min(MAX_POLL, max(MIN_POLL, interval * 1.5))

            await asyncio.sleep(interval + random.uniform(0, JITTER_MAX))

    def start(self):
        self._task = asyncio.create_task(self._poll_loop(), name="PhaseWatcher")

    async def stop(self):
        print("stopping PhaseWatcher")
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
