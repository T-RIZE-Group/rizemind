from eth_abi.abi import encode as abi_encode
from hexbytes import HexBytes
from web3 import Web3
from web3.contract.contract import ContractEvent
from web3.types import EventData, LogReceipt


def encode_with_selector(signature: str, arg_types: list[str], args: list) -> HexBytes:
    selector = Web3.keccak(text=signature)[:4]  # bytes4 selector
    encoded_args = abi_encode(arg_types, args)  # ABI-encode args
    return HexBytes(selector + encoded_args)


def decode_events_from_tx(
    *,
    deploy_tx: HexBytes,
    event: ContractEvent,
    w3: Web3,
) -> list[EventData]:
    """
    Return decoded instances of `event` found in `deploy_tx`, filtering by contract address and topic0.

    Parameters
    ----------
    deploy_tx : HexBytes
        The transaction receipt (from `w3.eth.wait_for_transaction_receipt`).
    event : ContractEvent
        Bound event class, e.g. `my_contract.events.AccessControlInstanceCreated`.

    Returns
    -------
    List[EventData]
        Decoded event objects (each has `.args` with decoded fields).
    """
    topic0 = Web3.to_bytes(hexstr=event.topic)
    contract_addr = event.address.lower()
    receipt = w3.eth.wait_for_transaction_receipt(deploy_tx)
    # Filter logs by address + topic0 to avoid MismatchedABI warnings.
    candidate_logs: list[LogReceipt] = [
        log
        for log in receipt["logs"]
        if log["address"].lower() == contract_addr
        and len(log.get("topics", [])) > 0
        and log["topics"][0] == topic0
    ]

    # Decode only the matching logs
    return [event().process_log(log) for log in candidate_logs]
