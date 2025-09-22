import os
from pathlib import Path
from typing import Any, Unpack

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.erc.erc5267.erc5267 import ERC5267
from rizemind.contracts.has_account import HasAccount
from web3.contract import Contract

abi_v1_0_0 = load_abi(Path(os.path.dirname(__file__)) / "./abi_1_0_0.json")


class SwarmV1InitializeParams:
    """Parameters for initializing SwarmV1 contract."""

    def __init__(
        self,
        name: str,
        initial_trainer_selector: ChecksumAddress,
        initial_evaluator_selector: ChecksumAddress,
        initial_contribution_calculator: ChecksumAddress,
        initial_access_control: ChecksumAddress,
        initial_compensation: ChecksumAddress,
        training_phase_configuration: dict[str, Any],
        evaluation_phase_configuration: dict[str, Any],
    ):
        self.name = name
        self.initial_trainer_selector = initial_trainer_selector
        self.initial_evaluator_selector = initial_evaluator_selector
        self.initial_contribution_calculator = initial_contribution_calculator
        self.initial_access_control = initial_access_control
        self.initial_compensation = initial_compensation
        self.training_phase_configuration = training_phase_configuration
        self.evaluation_phase_configuration = evaluation_phase_configuration


class SwarmV1(BaseContract, HasAccount):
    """SwarmV1 contract bindings for swarm coordination and training lifecycle."""

    abi_versions: dict[str, list[dict]] = {"swarm-v1.0.0": abi_v1_0_0}

    def __init__(self, contract: Contract, *, account: BaseAccount | None = None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    @staticmethod
    def from_address(
        *, account: BaseAccount | None = None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "SwarmV1":
        """Create SwarmV1 instance from contract address."""
        erc5267 = ERC5267.from_address(**kwargs)
        domain = erc5267.get_eip712_domain()
        model_abi = SwarmV1.get_abi(domain.version)
        return SwarmV1(contract_factory(**kwargs, abi=model_abi), account=account)

    @staticmethod
    def get_abi(version: str) -> list[dict]:
        """Get ABI for specific version."""
        if version in SwarmV1.abi_versions:
            return SwarmV1.abi_versions[version]
        raise ValueError(f"Version {version} not supported")

    def initialize(self, params: SwarmV1InitializeParams) -> HexBytes:
        """Initialize the SwarmV1 contract."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.initialize(
                (
                    params.name,
                    params.initial_trainer_selector,
                    params.initial_evaluator_selector,
                    params.initial_contribution_calculator,
                    params.initial_access_control,
                    params.initial_compensation,
                    params.training_phase_configuration,
                    params.evaluation_phase_configuration,
                )
            ),
            from_account=account,
        )

    def can_train(self, trainer: ChecksumAddress, round_id: int) -> bool:
        """Check if a trainer can participate in a specific round."""
        return self.contract.functions.canTrain(trainer, round_id).call()

    def can_evaluate(self, evaluator: ChecksumAddress, round_id: int) -> bool:
        """Check if an evaluator can participate in a specific round."""
        return self.contract.functions.canEvaluate(evaluator, round_id).call()

    def update_trainer_selector(
        self, new_trainer_selector: ChecksumAddress
    ) -> HexBytes:
        """Update the trainer selector (aggregator only)."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.updateTrainerSelector(new_trainer_selector),
            from_account=account,
        )

    def update_evaluator_selector(
        self, new_evaluator_selector: ChecksumAddress
    ) -> HexBytes:
        """Update the evaluator selector (aggregator only)."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.updateEvaluatorSelector(
                new_evaluator_selector
            ),
            from_account=account,
        )

    def distribute(
        self, round_id: int, trainers: list[ChecksumAddress], contributions: list[int]
    ) -> HexBytes:
        """Distribute rewards to trainers (aggregator only)."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.distribute(round_id, trainers, contributions),
            from_account=account,
        )

    def start_training_round(self) -> HexBytes:
        """Start a new training round (aggregator only)."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.startTrainingRound(),
            from_account=account,
        )

    def can_start_training_round(self) -> bool:
        """Check if the swarm can start a training round."""
        account = self.get_account()
        return self.simulate(
            tx_fn=self.contract.functions.startTrainingRound(),
            from_account=account,
        )

    def register_round_contribution(
        self, round_id: int, model_hash: HexBytes
    ) -> HexBytes:
        """Register a contribution for the current round (trainer only)."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.registerRoundContribution(
                round_id, model_hash
            ),
            from_account=account,
        )

    def register_for_round_evaluation(self, round_id: int) -> HexBytes:
        """Register for round evaluation (evaluator only)."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.registerForRoundEvaluation(round_id),
            from_account=account,
        )

    def register_evaluation(
        self,
        round_id: int,
        eval_id: int,
        set_id: int,
        model_hash: HexBytes,
        result: int,
    ) -> HexBytes:
        """Register an evaluation result."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.registerEvaluation(
                round_id, eval_id, set_id, model_hash, result
            ),
            from_account=account,
        )

    def claim_reward(self, round_id: int, trainer: ChecksumAddress) -> HexBytes:
        """Claim rewards for a completed round."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.claimReward(round_id, trainer),
            from_account=account,
        )

    def set_certificate(self, cert_id: HexBytes, value: bytes) -> HexBytes:
        """Set a certificate value (aggregator only)."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.setCertificate(cert_id, value),
            from_account=account,
        )

    # Getter methods for SwarmCore functionality
    def get_trainer_selector(self) -> ChecksumAddress:
        """Get the current trainer selector contract address."""
        return self.contract.functions.getTrainerSelector().call()

    def get_evaluator_selector(self) -> ChecksumAddress:
        """Get the current evaluator selector contract address."""
        return self.contract.functions.getEvaluatorSelector().call()

    def get_contribution_calculator(self) -> ChecksumAddress:
        """Get the current contribution calculator contract address."""
        return self.contract.functions.getContributionCalculator().call()

    def get_access_control(self) -> ChecksumAddress:
        """Get the current access control contract address."""
        return self.contract.functions.getAccessControl().call()

    def get_compensation(self) -> ChecksumAddress:
        """Get the current compensation contract address."""
        return self.contract.functions.getCompensation().call()
