import os
from pathlib import Path
from typing import Unpack, cast

from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from hexbytes import HexBytes
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount
from web3 import Web3
from web3.contract import Contract

abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class CertificateRegistry(HasAccount, BaseContract):
    def __init__(self, contract: Contract, account: BaseAccount | None = None):
        HasAccount.__init__(self, account=account)
        BaseContract.__init__(self, contract=contract)

    @staticmethod
    def from_address(
        account: BaseAccount | None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "CertificateRegistry":
        return CertificateRegistry(contract_factory(**kwargs, abi=abi), account=account)

    def get_certificate(self, cert_id: str) -> bytes:
        return self.contract.functions.getCertificate(self.hash_id(cert_id)).call()

    def set_certificate(self, cert_id: str, value: bytes) -> HexBytes:
        account = self.get_account()
        address = account.address
        tx = self.contract.functions.setCertificate(
            self.hash_id(cert_id), value
        ).build_transaction(
            {"from": address, "nonce": self.w3.eth.get_transaction_count(address)}
        )
        signed_tx = account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        assert tx_receipt["status"] != 0, "setCertificate returned an error"
        return tx_hash

    def hash_id(self, cert_id: str) -> HexBytes:
        return Web3.keccak(text=cert_id)
