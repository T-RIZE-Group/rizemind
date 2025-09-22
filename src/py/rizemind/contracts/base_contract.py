from typing import Any, TypedDict, Unpack, cast

from eth_abi.abi import decode
from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from eth_typing import ChecksumAddress, HexStr
from hexbytes import HexBytes
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractCustomError, ContractLogicError

from rizemind.contracts.abi_helper import PANIC_CODES, error_registry
from rizemind.exception.base_exception import RizemindException
from rizemind.exception.contract_execution_exception import RizemindContractError


class FromAddressKwargs(TypedDict):
    w3: Web3
    address: ChecksumAddress


class FactoryKwargs(TypedDict):
    w3: Web3
    address: ChecksumAddress
    abi: Any


def contract_factory(**kwargs: Unpack[FactoryKwargs]) -> Contract:
    return kwargs["w3"].eth.contract(
        address=kwargs["address"],
        abi=kwargs["abi"],
    )


class RizemindContractException(RizemindException):
    def __init__(self, code: str, message: str):
        super().__init__(code=code, message=message)


class BaseContract:
    contract: Contract

    def __init__(self, *, contract: Contract):
        self.contract = contract

    @property
    def w3(self) -> Web3:
        return self.contract.w3

    @property
    def address(self) -> ChecksumAddress:
        return self.contract.address

    def send(self, *, tx_fn: Any, from_account: BaseAccount) -> HexBytes:
        try:
            tx: TransactionDictType = tx_fn.build_transaction(
                self.get_transaction_context(from_account=from_account)
            )
            signed_tx = from_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        except (ContractLogicError, ContractCustomError) as e:
            self._raise_parsed_error(e)
        return tx_hash  # pyright: ignore[reportPossiblyUnboundVariable]

    def simulate(self, *, tx_fn: Any, from_account: BaseAccount) -> Any:
        try:
            return tx_fn.call(self.get_transaction_context(from_account=from_account))
        except (ContractLogicError, ContractCustomError) as e:
            self._raise_parsed_error(e)

    def _raise_parsed_error(self, e: ContractLogicError | ContractCustomError) -> Any:
        args: dict[str, str] = {}
        if isinstance(e.data, dict):
            error = {"name": "UnknownError", "inputs": []}
            args = e.data
        elif isinstance(e.data, str):
            data = Web3.to_bytes(hexstr=cast(HexStr, e.data))
            selector = data[0:4]
            error = error_registry.get(selector)
            input_types = [input.get("type", "") for input in error.get("inputs", [])]
            if input_types and len(data) > 4:
                decoded_args = decode(input_types, data[4:])
                args = {
                    inp.get("name", f"arg_{i}"): decoded_args[i]
                    for i, inp in enumerate(error.get("inputs", []))
                    if i < len(decoded_args)
                }
                if error.get("name", "") == "Panic":
                    args["error"] = PANIC_CODES[decoded_args[0]]
            else:
                args = {}
        else:
            raise ValueError("Unknown error data type")
        raise RizemindContractError(
            name=error.get("name", ""),
            error_args=args,
        ) from e

    def get_transaction_context(
        self, *, from_account: BaseAccount
    ) -> TransactionDictType:
        return {
            "from": from_account.address,
            "nonce": self.w3.eth.get_transaction_count(from_account.address),
        }
