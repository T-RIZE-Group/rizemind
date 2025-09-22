# async_runtime.py
import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future


class AsyncRuntime:
    """Owns an asyncio loop running forever on a background thread."""

    def __init__(self, name: str) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name=name, daemon=True
        )
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            print(f"Warning: AsyncRuntime thread didn't stop within {timeout}s")
        self._started = False
        self._started = False

    def submit(self, coro: Coroutine) -> Future:
        """Schedule a coroutine on the loop; returns concurrent.futures.Future."""
        if not self._started:
            self.start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop
