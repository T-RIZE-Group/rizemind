from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from rizemind.contracts.access_control.base_access_control.base_access_control import (
    BaseAccessControl,
)
from rizemind.contracts.compensation.simple_mint_compensation.simple_mint_compensation import (
    SimpleMintCompensation,
)
from rizemind.contracts.contribution.contribution_calculator.contribution_calculator import (
    ContributionCalculator,
)
from rizemind.contracts.erc.erc5267.erc5267 import (
    ERC5267,
    EIP712Domain,
)
from rizemind.contracts.scheduling.task_assigment.task_assignment import TaskAssignment
from rizemind.contracts.swarm.constants import CONTRIBUTION_DECIMALS
from rizemind.contracts.swarm.contributions.trainers_contributed import (
    TrainersContributedEventHelper,
)
from rizemind.contracts.swarm.registry.certificate_registry.certificate_registry import (
    CertificateRegistry,
)
from rizemind.contracts.swarm.registry.evaluator_registry.round_evaluator_registry import (
    RoundEvaluatorRegistry,
)
from rizemind.contracts.swarm.registry.swarm_core.swarm_core import SwarmCore
from rizemind.contracts.swarm.registry.trainer_registry.round_trainer_registry import (
    RoundTrainerRegistry,
)
from rizemind.contracts.swarm.swarm_v1.swarm_v1 import SwarmV1
from rizemind.contracts.swarm.training.base_training_phase.base_training_phase import (
    BaseTrainingPhases,
)
from rizemind.contracts.swarm.training.base_training_phase.phases import get_phase_name
from rizemind.contracts.swarm.training.round_training import RoundSummary, RoundTraining
from rizemind.exception.base_exception import RizemindException
from rizemind.exception.contract_execution_exception import RizemindContractError
from rizemind.swarm.certificate.certificate import Certificate, CompressedCertificate
from web3 import Web3
from web3.types import TxParams


class SwarmException(RizemindException):
    def __init__(self, message: str):
        super().__init__(code="swarm_exception", message=message)


class Swarm:
    address: ChecksumAddress
    access_control: BaseAccessControl
    training: RoundTraining
    swarm: SwarmV1
    swarm_core: SwarmCore
    contribution: TrainersContributedEventHelper
    contribution_calculator: ContributionCalculator
    compensation: SimpleMintCompensation
    erc5267: ERC5267
    certificates: CertificateRegistry
    phases: BaseTrainingPhases
    evaluator_registry: RoundEvaluatorRegistry
    trainer_registry: RoundTrainerRegistry
    task_assignement: TaskAssignment
    w3: Web3

    def __init__(
        self, *, w3: Web3, address: ChecksumAddress, account: BaseAccount | None
    ) -> None:
        self.address = address
        self.w3 = w3

        self.swarm = SwarmV1.from_address(w3=w3, address=address, account=account)
        self.swarm_core = SwarmCore.from_address(w3=w3, address=address)

        access_control_address = self.swarm_core.get_access_control()
        self.access_control = BaseAccessControl.from_address(
            w3=w3, address=access_control_address
        )
        self.contribution_calculator = ContributionCalculator.from_address(
            w3=w3, address=self.swarm_core.get_contribution_calculator()
        )
        self.compensation = SimpleMintCompensation.from_address(
            w3=w3, address=self.swarm_core.get_compensation(), account=account
        )
        self.training = RoundTraining.from_address(
            w3=w3, address=address, account=account
        )
        self.contribution = TrainersContributedEventHelper.from_address(
            w3=w3, address=address
        )
        self.erc5267 = ERC5267.from_address(w3=w3, address=address)
        self.certificates = CertificateRegistry.from_address(
            w3=w3, address=address, account=account
        )
        self.phases = BaseTrainingPhases.from_address(
            w3=w3, address=address, account=account
        )
        self.evaluator_registry = RoundEvaluatorRegistry.from_address(
            w3=w3, address=address, account=account
        )
        self.trainer_registry = RoundTrainerRegistry.from_address(
            w3=w3, address=address, account=account
        )
        self.task_assignement = TaskAssignment.from_address(
            w3=w3, address=address, account=account
        )

    def can_train(self, trainer: ChecksumAddress, round_id: int) -> bool:
        return self.swarm.can_train(trainer, round_id)

    def can_evaluate(self, evaluator: ChecksumAddress, round_id: int) -> bool:
        return self.swarm.can_evaluate(evaluator, round_id)

    def can_start_training_round(self) -> bool:
        try:
            self.swarm.can_start_training_round()
            return True
        except RizemindContractError:
            return False

    def is_training(self) -> bool:
        return self.phases.is_training()

    def is_evaluation(self) -> bool:
        return self.phases.is_evaluation()

    def is_idle(self) -> bool:
        return self.phases.is_idle()

    def get_current_phase(self) -> str:
        phase_id = self.phases.get_updated_phase()
        return get_phase_name(phase_id)

    def update_phase(self) -> bool:
        hash = self.phases.update_phase()
        receipt = self.w3.eth.wait_for_transaction_receipt(hash)
        if receipt["status"] != 1:
            tx = self.w3.eth.get_transaction(hash)
            replay: TxParams = {
                "to": tx["to"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                "from": tx["from"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                "value": tx["value"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                "data": tx["input"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
            }
            try:
                w3 = self.w3
                # replay in the original context (block n-1)
                w3.eth.call(replay, tx["blockNumber"] - 1)  # pyright: ignore[reportTypedDictNotRequiredAccess]
            except Exception as e:
                # e.args[0] contains "execution reverted: <reason>"
                # on newer web3.py, e.data has raw error bytes
                print("Reason:", str(e))
                raise SwarmException("update_phase failed to confirm (replay)") from e
            raise SwarmException(
                "update_phase failed to confirm ( didn't fail in replay)"
            )
        return True

    def is_aggregator(self, trainer: ChecksumAddress, round_id: int) -> bool:
        return self.access_control.is_aggregator(trainer)

    def distribute(
        self, round_id: int, trainer_scores: list[tuple[ChecksumAddress, float]]
    ) -> str:
        trainers = [trainer for trainer, _ in trainer_scores]
        contributions = [
            int(contribution * 10**CONTRIBUTION_DECIMALS)
            for _, contribution in trainer_scores
        ]
        return self.swarm.distribute(round_id, trainers, contributions).to_0x_hex()

    def get_current_round(self) -> int:
        return self.training.current_round()

    def next_round(
        self,
        round_id: int,
        n_trainers: int,
        model_score: float,
        total_contributions: float,
    ) -> str:
        return self.training.next_round(
            round_id=round_id,
            n_trainers=n_trainers,
            model_score=model_score,
            total_contributions=total_contributions,
        ).to_0x_hex()

    def current_round(self) -> int:
        return self.training.current_round()

    def get_latest_contribution(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> float | None:
        return self.contribution.get_latest_contribution(
            trainer, from_block=from_block, to_block=to_block
        )

    def get_latest_contribution_log(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> dict | None:
        return self.contribution.get_latest_contribution_log(
            trainer, from_block=from_block, to_block=to_block
        )

    def get_eip712_domain(self) -> EIP712Domain:
        return self.erc5267.get_eip712_domain()

    def get_last_contributed_round_summary(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> RoundSummary | None:
        """
        Returns the summary of the latest FINISHED round the trainer has contributed to
        """
        latest_contribution = self.get_latest_contribution_log(
            trainer, from_block, to_block
        )

        if latest_contribution is None:
            return None
        return self.training.get_round_at(latest_contribution["blockNumber"])

    def set_certificate(self, id: str, cert: Certificate) -> bool:
        compressed_cert = cert.get_compressed_bytes()
        data = compressed_cert.serialize()
        hash = self.certificates.set_certificate(id, data)
        self.w3.eth.wait_for_transaction_receipt(hash)
        return True

    def get_certificate(self, id: str) -> Certificate | None:
        data = self.certificates.get_certificate(id)
        if len(data) == 0:
            return None
        compressed_certificate = CompressedCertificate.deserialize(data)
        return Certificate.from_compressed(compressed_certificate)

    def start_training_round(self) -> bool:
        hash = self.swarm.start_training_round()
        receipt = self.w3.eth.wait_for_transaction_receipt(hash)
        if receipt["status"] != 1:
            raise SwarmException("start_training_round failed to confirm")
        return True

    def register_round_contribution(self, round_id: int, model_hash: HexBytes) -> bool:
        hash = self.swarm.register_round_contribution(round_id, model_hash)
        receipt = self.w3.eth.wait_for_transaction_receipt(hash)
        if receipt["status"] != 1:
            tx = self.w3.eth.get_transaction(hash)
            replay: TxParams = {
                "to": tx["to"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                "from": tx["from"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                "value": tx["value"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                "data": tx["input"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
            }
            try:
                w3 = self.w3
                # replay in the original context (block n-1)
                w3.eth.call(replay, tx["blockNumber"])  # pyright: ignore[reportTypedDictNotRequiredAccess]
            except Exception as e:
                # e.args[0] contains "execution reverted: <reason>"
                # on newer web3.py, e.data has raw error bytes
                print("Reason:", str(e))
                raise SwarmException(
                    "register_round_contribution failed to confirm (replay)"
                ) from e
            raise SwarmException(
                "register_round_contribution failed to confirm ( didn't fail in replay)"
            )
        return True

    def register_for_round_evaluation(self, round_id: int) -> bool:
        hash = self.swarm.register_for_round_evaluation(round_id)
        receipt = self.w3.eth.wait_for_transaction_receipt(hash)
        if receipt["status"] != 1:
            raise SwarmException("register_for_round_evaluation failed to confirm")
        return True

    def register_evaluation(
        self,
        round_id: int,
        eval_id: int,  # task ID
        set_id: int,  # bitvector
        model_hash: HexBytes,
        result: int,
    ) -> bool:
        hash = self.swarm.register_evaluation(
            round_id, eval_id, set_id, model_hash, result
        )
        receipt = self.w3.eth.wait_for_transaction_receipt(hash)
        if receipt["status"] != 1:
            raise SwarmException("register_for_round_evaluation failed to confirm")
        return True
