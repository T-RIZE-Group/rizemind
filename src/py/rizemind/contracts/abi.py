from eth_abi.abi import encode as abi_encode
from hexbytes import HexBytes
from web3 import Web3


def encode_with_selector(signature: str, arg_types: list[str], args: list) -> HexBytes:
    selector = Web3.keccak(text=signature)[:4]  # bytes4 selector
    encoded_args = abi_encode(arg_types, args)  # ABI-encode args
    return HexBytes(selector + encoded_args)
