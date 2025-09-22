import json
from pathlib import Path
from typing import Any

from eth_typing import ABIError, ABIEvent, HexStr
from eth_utils.abi import (
    event_abi_to_log_topic,
    filter_abi_by_type,
    function_abi_to_4byte_selector,
)
from web3 import Web3

from rizemind.exception.base_exception import RizemindException

ERROR_SELECTOR = Web3.keccak(text="Error(string)")[:4]
PANIC_SELECTOR = Web3.keccak(text="Panic(uint256)")[:4]
PANIC_CODES = {
    0x01: "assert(false)",
    0x11: "overflow/underflow",
    0x12: "div/mod by zero",
    0x21: "enum OOB",
    0x22: "invalid storage bytes",
    0x31: "pop empty array",
    0x32: "index OOB",
    0x41: "memory overflow",
    0x51: "uninit function pointer",
}


class ErrorRegistry:
    errors: dict[bytes, ABIError]

    def __init__(self):
        self.errors = {}
        self.errors[ERROR_SELECTOR] = {
            "name": "Error",
            "inputs": [{"name": "error", "type": "string"}],
            "type": "error",
        }
        self.errors[PANIC_SELECTOR] = {
            "name": "Panic",
            "inputs": [{"name": "error", "type": "uint256"}],
            "type": "error",
        }

    def register(self, error: ABIError):
        selector = function_abi_to_4byte_selector(error)
        self.errors[selector] = error

    def get(self, selector: bytes) -> ABIError:
        return self.errors[selector]


class EventsRegistry:
    events: dict[HexStr, ABIEvent]

    def __init__(self):
        self.events = {}

    def register(self, event: ABIEvent):
        topic = event_abi_to_log_topic(event)
        self.events[Web3.to_hex(topic)] = event

    def get(self, topic: bytes) -> ABIEvent:
        return self.events[Web3.to_hex(topic)]


error_registry = ErrorRegistry()
events_registry = EventsRegistry()


class AbiNotFoundError(RizemindException):
    def __init__(self, path: Path):
        super().__init__(code="abi_not_found", message=f"ABI not found: {path}")


type Abi = Any


def load_abi(path: Path) -> Abi:
    """
    Load and validate an ABI
    """
    if not path.exists():
        raise AbiNotFoundError(path)

    with open(path, encoding="utf-8") as f:
        abi = json.load(f)

    Web3().eth.contract(abi=abi)

    errors = filter_abi_by_type("error", abi)
    for error in errors:
        error_registry.register(error)

    events = filter_abi_by_type("event", abi)
    for event in events:
        events_registry.register(event)

    return abi
