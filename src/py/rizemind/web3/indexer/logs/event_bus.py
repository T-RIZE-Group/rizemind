import asyncio
import contextlib
from dataclasses import dataclass

from eth_typing import ABIEvent
from eth_utils.abi import event_abi_to_log_topic
from rizemind.contracts.abi_helper import EventsRegistry
from rizemind.web3.indexer.logs.typing import EventEnvelope, Handler, Predicate
from web3 import AsyncWeb3, Web3
from web3.contract.base_contract import BaseContractEvent
from web3.types import EventData, LogReceipt


@dataclass(frozen=True)
class Filter:
    topic0: bytes
    contract: str | None


@dataclass
class Subscription:
    filter: Filter
    handler: Handler


class EventBus:
    _w3: AsyncWeb3
    _events_registry: EventsRegistry
    _lock: asyncio.Lock
    _next_token: int
    _tokens: dict[int, tuple[list[Subscription | None], int]]
    _all: list[Subscription | None]
    _topic0_subscription: dict[bytes, list[Subscription | None]]

    def __init__(self, w3: AsyncWeb3):
        self._w3 = w3
        self.topic0_subscription = {}
        self._all = []
        self._tokens = {}
        self._next_token = 1
        self._lock = asyncio.Lock()
        self._events_registry = EventsRegistry()

    async def on(
        self,
        *,
        event: ABIEvent,  # ABI event entry
        handler: Handler,  # async def handler(evt: dict): ...
        contract: str | None = None,  # checksum or None
    ) -> int:
        self._events_registry.register(event)
        topic0 = event_abi_to_log_topic(event)
        filter = Filter(topic0=topic0, contract=self._normalize_addr(contract))
        sub = Subscription(filter=filter, handler=handler)
        async with self._lock:
            topic_subscription = self.topic0_subscription.setdefault(topic0, [])
            topic_subscription.append(sub)
            token = self._next_token
            self._next_token += 1
            self._tokens[token] = (topic_subscription, len(topic_subscription) - 1)
        return token

    async def on_any(self, handler: Handler) -> int:
        sub = Subscription(Filter(b"*", None), handler)
        async with self._lock:
            self._all.append(sub)
            token = self._next_token
            self._next_token += 1
            self._tokens[token] = (self._all, len(self._all) - 1)
        return token

    async def unsubscribe(self, token: int) -> None:
        async with self._lock:
            entry = self._tokens.pop(token, None)
            if not entry:
                return
            lst, idx = entry
            if 0 <= idx < len(lst):
                lst[idx] = None  # leave a hole; O(1) delete

    async def publish_raw_log(self, log: LogReceipt) -> None:
        topics = log.get("topics") or []
        if not topics:
            return
        topic0 = topics[0]

        async with self._lock:
            subs = list(self._topic0_subscription.get(topic0, []))
            any_subs = list(self._all)

        if not subs and not any_subs:
            return

        addr = self._normalize_addr(log.get("address"))
        matched: list[Subscription] = []
        for subscription in subs:
            if subscription is None:
                continue
            if subscription.filter.contract and addr != subscription.filter.contract:
                continue
            matched.append(subscription)

        if not matched and not any_subs:
            return

        event_abi = self._events_registry.get(topic0)
        if not event_abi:
            return

        try:
            event = BaseContractEvent(abi=event_abi)
            event_data: EventData = event.process_log(log)
            envelope = self._envelope(topic0, event_abi, event_data)
        except Exception:
            return

        to_call: list[Handler] = [s.handler for s in matched if s is not None] + [
            s.handler for s in any_subs if s is not None
        ]
        if to_call:
            await asyncio.gather(
                *(self._safe_call(handlers, envelope) for handlers in to_call),
                return_exceptions=True,
            )

    async def wait_for_async(
        self,
        *,
        event: ABIEvent,
        contract: str | None,
        predicate: Predicate,
        timeout: float | None = None,
    ) -> EventEnvelope:
        """Await the first event matching (event, contract, predicate)."""

        loop = asyncio.get_running_loop()
        done: asyncio.Future[EventEnvelope] = loop.create_future()
        token: int | None = None

        async def _handler(envelope: EventEnvelope) -> None:
            nonlocal token
            try:
                if not done.done() and predicate(envelope):
                    done.set_result(envelope)
            finally:
                # best-effort cleanup; okay if already unsubscribed
                if token is not None:
                    with contextlib.suppress(Exception):
                        await self.unsubscribe(token)

        # register temp subscription
        token = await self.on(event=event, handler=_handler, contract=contract)

        try:
            if timeout is None:
                return await done
            return await asyncio.wait_for(done, timeout=timeout)
        finally:
            # if timed out/cancelled and still subscribed, remove it
            if not done.done() and token is not None:
                with contextlib.suppress(Exception):
                    await self.unsubscribe(token)

    @staticmethod
    def _normalize_addr(addr: str | None) -> str | None:
        if not addr:
            return None
        try:
            return Web3.to_checksum_address(addr)
        except Exception:
            return None

    @staticmethod
    async def _safe_call(handler: Handler, evt: EventEnvelope) -> None:
        try:
            await handler(evt)
        except Exception:
            pass

    @staticmethod
    def _envelope(topic0: bytes, abi: ABIEvent, decoded: EventData) -> EventEnvelope:
        txh = decoded["transactionHash"]
        bh = decoded.get("blockHash")
        return {
            "name": abi.get("name", "<unknown>"),
            "signature": topic0,
            "abi": abi,
            "address": Web3.to_checksum_address(decoded["address"]),
            "args": decoded["args"],
            "blockNumber": decoded["blockNumber"],
            "transactionHash": txh.hex(),
            "logIndex": decoded["logIndex"],
            "blockHash": bh.hex(),
            "removed": decoded.get("removed", False),
        }
