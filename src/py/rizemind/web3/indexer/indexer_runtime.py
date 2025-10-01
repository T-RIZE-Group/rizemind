from concurrent.futures import Future, TimeoutError

from rizemind.workflow.async_runtime import AsyncRuntime

from .indexer import Indexer


class IndexerRuntime:
    """Sync facade around the async Indexer running on its own loop/thread."""

    def __init__(self, indexer: Indexer, async_runtime: AsyncRuntime) -> None:
        self._rt = async_runtime
        self._indexer = indexer
        self._future: Future | None = None

    def start(self) -> None:
        """Start the loop thread and schedule indexer.run() (non-blocking)."""
        if self._future is None or self._future.done():
            self._future = self._rt.submit(self._indexer.run())

    def stop(self, timeout: float = 10.0) -> None:
        """Request graceful stop and wait for completion; then stop the loop."""
        if self._future is None:
            return
        # Signal stop on the loop thread
        self._rt.loop.call_soon_threadsafe(self._indexer.request_stop)
        try:
            self._future.result(timeout=timeout)  # re-raises exceptions if any
        except TimeoutError:
            # Last resort: cancel the task by stopping the loop
            pass
