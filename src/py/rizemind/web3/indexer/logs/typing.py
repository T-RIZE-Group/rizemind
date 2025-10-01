from collections.abc import Awaitable, Callable
from typing import TypedDict

from eth_typing import ABIEvent


class EventEnvelope(TypedDict):
    name: str
    signature: bytes
    abi: ABIEvent
    address: str
    args: dict
    blockNumber: int
    transactionHash: str
    logIndex: int
    blockHash: str | None
    removed: bool


Predicate = Callable[[EventEnvelope], bool]
Handler = Callable[[EventEnvelope], Awaitable[None]]
