import asyncio

from eth_typing import ABIEvent, ChecksumAddress
from rizemind.web3.indexer.logs.typing import Predicate

from .event_bus import EventBus, EventEnvelope


class SyncEventBus:
    """Blocking facade for EventBus, executed on a specific asyncio loop."""

    def __init__(self, bus: EventBus, loop: asyncio.AbstractEventLoop) -> None:
        self._bus = bus
        self._loop = loop

    def wait_for(
        self,
        *,
        event: ABIEvent,
        contract: ChecksumAddress | None,
        predicate: Predicate,
        timeout: float | None = None,
    ) -> EventEnvelope:
        """Block the calling thread until a matching event arrives (or timeout)."""
        coro = self._bus.wait_for_async(
            event=event,
            contract=contract,
            predicate=predicate,
            timeout=timeout,
        )
        # Run coroutine on the indexer loop and block here
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # Let the coroutine enforce timeout via asyncio.wait_for; no need to pass again
        return fut.result()
