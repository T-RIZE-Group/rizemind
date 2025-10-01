from typing import Any, cast

from eth_abi.abi import decode
from eth_typing import ABIError, HexStr
from rizemind.contracts.abi_helper import PANIC_CODES, error_registry
from rizemind.exception.contract_execution_exception import RizemindContractError
from web3 import Web3
from web3._utils.rpc_abi import (
    RPC,
)
from web3.middleware import FormattingMiddlewareBuilder


def _parse_error(hex_data: str) -> tuple[ABIError, dict[str, Any]]:
    """
    Returns (error_abi, args_dict).
    error_abi: {"name": str, "inputs": [{"name":..., "type":...}, ...]}
    """
    data = Web3.to_bytes(hexstr=cast(HexStr, hex_data))
    selector = data[0:4]
    error = error_registry.get(selector)
    inputs = error.get("inputs", [])
    types = [i.get("type", "") for i in inputs]
    args: dict[str, Any] = {}
    if types and len(data) > 4:
        decoded = decode(types, data[4:])
        args = {
            inp.get("name", f"arg_{i}"): decoded[i]
            for i, inp in enumerate(inputs)
            if i < len(decoded)
        }
        if error.get("name") == "Panic" and decoded:
            args["error"] = PANIC_CODES.get(decoded[0], "UnknownPanicCode")
    return error, args


def _extract_hex_error_data(err_data: Any) -> str | None:
    # Common shapes:
    # 1) "0x08c379a0...."
    # 2) {"data": "0x..."} (geth / parity)
    # 3) {"<txhash>": {"error": {..., "data": "0x..."}}} (eth_sendRawTransaction failures)
    if isinstance(err_data, str):
        return err_data
    if isinstance(err_data, dict):
        if isinstance(err_data.get("data"), str):
            return err_data["data"]
        for v in err_data.values():
            if isinstance(v, dict) and isinstance(v.get("data"), str):
                return v["data"]
    return None


def _raise_rizemind_if_revert(error_obj: dict[str, Any]) -> dict[str, Any]:
    """
    error_formatters receive the JSON-RPC 'error' object.
    If we can decode a Solidity error, raise RizemindContractError.
    Otherwise, return the original error so Web3 can handle it.
    """
    hex_data = _extract_hex_error_data(error_obj.get("data"))
    if not hex_data:
        return error_obj  # unchanged; Web3 will raise its default error later

    error_abi, args = _parse_error(hex_data)
    raise RizemindContractError(
        name=error_abi.get("name", "UnknownError"),
        error_args=args,
    )


# ---------- the middleware you can register ----------

RizemindErrorsMiddleware = FormattingMiddlewareBuilder.build(  # pyright: ignore[reportCallIssue]
    # We target only the RPCs that surface EVM reverts
    error_formatters={
        RPC.eth_call: _raise_rizemind_if_revert,
        RPC.eth_sendRawTransaction: _raise_rizemind_if_revert,
    },
    # (optional) you could also map result_formatters if you wanted to massage successful results
    # result_formatters={}
)
