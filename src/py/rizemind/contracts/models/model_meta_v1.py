import math
from logging import INFO
from typing import Optional, Tuple, cast

from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from eth_typing import Address
from flwr.common.logger import log
from hexbytes import HexBytes
from rizemind.contracts.abi.model_v1 import model_abi_v1_0_0
from rizemind.contracts.access_control.FlAccessControl import FlAccessControl
from rizemind.contracts.models.constants import (
    CONTRIBUTION_DECIMALS,
    MODEL_SCORE_DECIMALS,
)
from rizemind.contracts.models.erc5267 import ERC5267
from rizemind.contracts.models.model_meta import ModelMeta
from web3 import Web3
from web3.contract import Contract


class ModelMetaV1(FlAccessControl, ModelMeta):
    account: Optional[BaseAccount]
    w3: Web3

    abi_versions: dict[str, list[dict]] = {"1.0.0": model_abi_v1_0_0}

    def __init__(self, model: Contract, account: Optional[BaseAccount], w3: Web3):
        FlAccessControl.__init__(self, model)
        ModelMeta.__init__(self, model)
        self.account = account
        self.w3 = w3

    def distribute(self, trainer_scores: list[tuple[Address, float]]) -> HexBytes:
        if self.account is None:
            raise Exception("No account connected")

        trainers = [trainer for trainer, _ in trainer_scores]
        contributions = [
            int(contribution * 10**CONTRIBUTION_DECIMALS)
            for _, contribution in trainer_scores
        ]

        address = self.account.address
        tx = self.model.functions.distribute(trainers, contributions).build_transaction(
            {"from": address, "nonce": self.w3.eth.get_transaction_count(address)}
        )
        signed_tx = self.account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        log(
            INFO,
            "distribute: trainers rewards distributed:",
        )
        log(INFO, "Reward (Address, Value):")
        for trainer, contribution in zip(trainers, contributions):
            log(INFO, "\t(%s, %s)", trainer, contribution)

        assert tx_receipt["status"] != 0, "distribute returned an error"
        return tx_hash

    def next_round(
        self,
        round_id: int,
        n_trainers: int,
        model_score: float,
        total_contributions: float,
    ) -> HexBytes:
        if self.account is None:
            raise Exception("No account connected")

        address = self.account.address
        round_summary_data: Tuple[int, int, int, int] = (
            round_id,
            n_trainers,
            math.floor(model_score * 10**MODEL_SCORE_DECIMALS),
            math.floor(total_contributions * 10**CONTRIBUTION_DECIMALS),
        )
        tx = self.model.functions.nextRound(round_summary_data).build_transaction(
            {"from": address, "nonce": self.w3.eth.get_transaction_count(address)}
        )
        signed_tx = self.account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        log(
            INFO,
            "next_round: marked round as finished",
        )

        assert tx_receipt["status"] != 0, "nextRound returned an error"
        return tx_hash

    @staticmethod
    def from_address(
        address: str, w3: Web3, account: Optional[BaseAccount] = None
    ) -> "ModelMetaV1":
        erc5267 = ERC5267.from_address(address, w3)
        domain = erc5267.get_eip712_domain()
        model_abi = ModelMetaV1.get_abi(domain.version)
        checksum_address = Web3.to_checksum_address(address)
        return ModelMetaV1(
            w3.eth.contract(address=checksum_address, abi=model_abi), account, w3
        )

    @staticmethod
    def get_abi(version: str) -> list[dict]:
        if version in ModelMetaV1.abi_versions:
            return ModelMetaV1.abi_versions[version]
        raise Exception(f"Version {version} not supported")
