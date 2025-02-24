from typing import Any, cast
from eth_typing import Address, ChecksumAddress
from rizemind.contracts.access_control.FlAccessControl import FlAccessControl
from rizemind.contracts.models.erc5267 import ERC5267
from rizemind.contracts.models.model_registry import ModelRegistry
from web3 import Web3
from eth_account.signers.base import BaseAccount
from web3.contract import Contract
from eth_account.types import TransactionDictType
from rizemind.contracts.abi.model_v1 import model_abi_v1_0_0


class ModelRegistryV1(FlAccessControl, ModelRegistry):
    account: BaseAccount
    w3: Web3

    abi_versions: dict[str, list[dict]] = {"1.0.0": model_abi_v1_0_0}

    def __init__(self, model: Contract, account: BaseAccount, w3: Web3):
        FlAccessControl.__init__(self, model)
        ModelRegistry.__init__(self, model)
        self.account = account
        self.w3 = w3

    def distribute(self, trainers: list[Address], contributions: list[Any]) -> bool:
        if self.account.address is None:
            raise Exception("no signer")
        address = ChecksumAddress(self.account.address)
        tx = self.model.functions.distribute(trainers, contributions).build_transaction(
            {"from": address, "nonce": self.w3.eth.get_transaction_count(address)}
        )
        signed_tx = self.account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_receipt["status"] == 0

    @staticmethod
    def from_address(address: str, account: BaseAccount, w3: Web3) -> "ModelRegistryV1":
        erc5267 = ERC5267.from_address(address, w3)
        domain = erc5267.get_eip712_domain()
        model_abi = ModelRegistryV1.get_abi(domain.version)
        checksum_address = Web3.to_checksum_address(address)
        return ModelRegistryV1(
            w3.eth.contract(address=checksum_address, abi=model_abi), account, w3
        )

    @staticmethod
    def get_abi(version: str) -> list[dict]:
        if version in ModelRegistryV1.abi_versions:
            return ModelRegistryV1.abi_versions[version]
        raise Exception(f"Version {version} not supported")
