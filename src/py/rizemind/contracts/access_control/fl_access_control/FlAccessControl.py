import os
from pathlib import Path
from typing import Unpack, cast

from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from eth_typing import ChecksumAddress
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount
from web3.contract import Contract

abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class FlAccessControl(BaseContract, HasAccount):
    def __init__(self, contract: Contract, *, account: BaseAccount | None = None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    @staticmethod
    def from_address(**kwargs: Unpack[FromAddressKwargs]) -> "FlAccessControl":
        return FlAccessControl(contract_factory(**kwargs, abi=abi))

    def is_trainer(self, address: str) -> bool:
        return self.contract.functions.isTrainer(address).call()

    def is_aggregator(self, address: str) -> bool:
        return self.contract.functions.isAggregator(address).call()

    def initialize(
        self, aggregator: ChecksumAddress, trainers: list[ChecksumAddress]
    ) -> str:
        account = self.get_account()
        tx = self.contract.functions.initialize(aggregator, trainers).build_transaction(
            {
                "from": account.address,
                "nonce": self.w3.eth.get_transaction_count(account.address),
            }
        )
        signed = account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()
