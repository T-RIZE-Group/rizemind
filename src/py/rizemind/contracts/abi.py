from eth_abi.abi import encode as abi_encode
from hexbytes import HexBytes
from web3 import Web3
from web3.constants import ADDRESS_ZERO
from web3.contract.contract import ContractEvent
from web3.types import EventData, LogReceipt


def encode_with_selector(signature: str, arg_types: list[str], args: list) -> HexBytes:
    selector = Web3.keccak(text=signature)[:4]  # bytes4 selector
    encoded_args = abi_encode(arg_types, args)  # ABI-encode args
    return HexBytes(selector + encoded_args)


def decode_events_from_tx(
    *,
    tx_hash: HexBytes,
    event: ContractEvent,
    w3: Web3,
) -> list[EventData]:
    """
    Decode all instances of the given `event` emitted in the transaction `tx_hash`.

    Parameters
    ----------
    tx_hash : HexBytes
        Transaction hash to wait on and inspect (from `w3.eth.wait_for_transaction_receipt`).
    event : ContractEvent
        Bound event class (e.g., `my_contract.events.AccessControlInstanceCreated`).
        Must have `.address` set (the emitting contract) and `.topic` (the event's topic0).
    w3 : Web3
        A Web3 instance.

    Returns
    -------
    list[EventData]
        Decoded event objects (each has `.args` with decoded fields).
    """
    # Resolve identifying filters
    topic0 = Web3.to_bytes(hexstr=event.topic)  # canonical bytes for topic0
    contract_address = Web3.to_checksum_address(event.address)

    # Fetch the transaction receipt once
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    # Pre-filter logs by contract address and topic0 to avoid MismatchedABI warnings
    candidate_logs: list[LogReceipt] = []
    for log in receipt["logs"]:
        if (
            Web3.to_checksum_address(log.get("address", ADDRESS_ZERO))
            != contract_address
        ):
            continue

        topics = log.get("topics", [])
        if not topics:
            continue
        if topics[0] != topic0:
            continue
        candidate_logs.append(log)

    # Decode the matching logs into EventData objects
    return [event().process_log(log) for log in candidate_logs]
