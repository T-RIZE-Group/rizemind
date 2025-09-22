import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from hexbytes import HexBytes
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount
from web3.contract import Contract

from .config import BaseEvaluationPhaseConfig, BaseTrainingPhaseConfig
from .phases import (
    EVALUATION_PHASE_ID,
    EVALUATOR_REGISTRATION_PHASE_ID,
    IDLE_PHASE_ID,
    TRAINING_PHASE_ID,
)

abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class BaseTrainingPhases(BaseContract, HasAccount):
    """BaseTrainingPhases contract bindings for managing training lifecycle phases."""

    def __init__(self, contract: Contract, *, account: BaseAccount | None = None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    @staticmethod
    def from_address(
        *, account: BaseAccount | None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "BaseTrainingPhases":
        """Create BaseTrainingPhases instance from contract address."""
        return BaseTrainingPhases(contract_factory(**kwargs, abi=abi), account=account)

    def get_current_phase(self) -> HexBytes:
        """Get the current phase of the training lifecycle."""
        return self.contract.functions.getCurrentPhase().call()

    def update_phase(self) -> HexBytes:
        """Update the phase based on current time and configuration."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.updatePhase(),
            from_account=account,
        )

    def get_updated_phase(self) -> HexBytes:
        """Get the updated phase of the training lifecycle."""
        account = self.get_account()
        return self.simulate(
            tx_fn=self.contract.functions.updatePhase(),
            from_account=account,
        )

    def is_training(self) -> bool:
        """Check if currently in training phase."""
        return self.contract.functions.isTraining().call()

    def is_evaluation(self) -> bool:
        """Check if currently in evaluation phase (includes evaluator registration)."""
        return self.contract.functions.isEvaluation().call()

    def is_idle(self) -> bool:
        """Check if currently in idle phase."""
        return self.contract.functions.isIdle().call()

    def is_phase(self, phase: HexBytes) -> bool:
        """Check if currently in a specific phase."""
        return self.contract.functions.isPhase(phase).call()

    # Convenience methods for specific phases
    def is_idle_phase(self) -> bool:
        """Check if currently in idle phase."""
        return self.is_phase(IDLE_PHASE_ID)

    def is_training_phase(self) -> bool:
        """Check if currently in training phase."""
        return self.is_phase(TRAINING_PHASE_ID)

    def is_evaluator_registration_phase(self) -> bool:
        """Check if currently in evaluator registration phase."""
        return self.is_phase(EVALUATOR_REGISTRATION_PHASE_ID)

    def is_evaluation_phase(self) -> bool:
        """Check if currently in evaluation phase."""
        return self.is_phase(EVALUATION_PHASE_ID)

    def get_training_phase_configuration(self) -> BaseTrainingPhaseConfig:
        """Get the training phase configuration."""
        config_data = self.contract.functions.getTrainingPhaseConfiguration().call()
        return BaseTrainingPhaseConfig(ttl=config_data[0])

    def get_evaluation_phase_configuration(self) -> BaseEvaluationPhaseConfig:
        """Get the evaluation phase configuration."""
        config_data = self.contract.functions.getEvaluationPhaseConfiguration().call()
        return BaseEvaluationPhaseConfig(
            ttl=config_data[0],
            registration_ttl=config_data[1],
        )
