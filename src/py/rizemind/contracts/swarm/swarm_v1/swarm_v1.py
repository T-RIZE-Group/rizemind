import os
from logging import INFO
from pathlib import Path
from typing import Unpack, cast

from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from eth_typing import ChecksumAddress, HexAddress
from flwr.common.logger import log
from hexbytes import HexBytes
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.erc.erc5267.erc5267 import ERC5267
from rizemind.contracts.has_account import HasAccount
from rizemind.contracts.swarm.constants import (
    CONTRIBUTION_DECIMALS,
)
from web3.contract import Contract

abi_v1_0_0 = load_abi(Path(os.path.dirname(__file__)) / "./abi_1_0_0.json")


class SwarmV1(BaseContract, HasAccount):
    abi_versions: dict[str, list[dict]] = {"swarm-v1.0.0": abi_v1_0_0}

    def __init__(self, contract: Contract, account: BaseAccount | None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    def distribute(
        self, trainer_scores: list[tuple[ChecksumAddress, float]]
    ) -> HexBytes:
        account = self.get_account()
        trainers = [trainer for trainer, _ in trainer_scores]
        contributions = [
            int(contribution * 10**CONTRIBUTION_DECIMALS)
            for _, contribution in trainer_scores
        ]

        address = account.address
        tx = self.contract.functions.distribute(
            trainers, contributions
        ).build_transaction(
            {"from": address, "nonce": self.w3.eth.get_transaction_count(address)}
        )
        signed_tx = account.sign_transaction(cast(TransactionDictType, tx))
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

    def can_train(self, trainer: HexAddress, round_id: int) -> bool:
        return self.contract.functions.canTrain(trainer, round_id).call()

    @staticmethod
    def from_address(
        *, account: BaseAccount | None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "SwarmV1":
        erc5267 = ERC5267.from_address(**kwargs)
        domain = erc5267.get_eip712_domain()
        model_abi = SwarmV1.get_abi(domain.version)
        return SwarmV1(contract_factory(**kwargs, abi=model_abi), account)

    @staticmethod
    def get_abi(version: str) -> list[dict]:
        if version in SwarmV1.abi_versions:
            return SwarmV1.abi_versions[version]
        raise Exception(f"Version {version} not supported")
