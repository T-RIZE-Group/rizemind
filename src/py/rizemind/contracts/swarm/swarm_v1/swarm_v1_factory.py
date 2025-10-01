import os
from logging import INFO
from pathlib import Path
from typing import cast

from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from eth_typing import ChecksumAddress
from flwr.common.logger import log
from hexbytes import HexBytes
from pydantic import BaseModel, Field
from rizemind.contracts.abi import decode_events_from_tx
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.access_control.base_access_control.base_access_control import (
    BaseAccessControlConfig,
)
from rizemind.contracts.compensation.simple_mint_compensation.simple_mint_compensation import (
    SimpleMintCompensationConfig,
)
from rizemind.contracts.contribution.contribution_calculator.contribution_calculator import (
    ContributionCalculatorConfig,
)
from rizemind.contracts.deployment import DeployedContract
from rizemind.contracts.local_deployment import load_forge_artifact
from rizemind.contracts.sampling.always_sampled import AlwaysSamplesSelectorConfig
from rizemind.contracts.sampling.random_sampling import RandomSamplingSelectorConfig
from rizemind.contracts.swarm.training.base_training_phase.config import (
    BaseEvaluationPhaseConfig,
    BaseTrainingPhaseConfig,
)
from rizemind.web3.chains import RIZENET_TESTNET_CHAINID
from web3 import Web3
from web3.eth import Contract

available_selectors = [
    AlwaysSamplesSelectorConfig,
    RandomSamplingSelectorConfig,
]

type AvailableSelector = AlwaysSamplesSelectorConfig | RandomSamplingSelectorConfig

available_compensations = [
    SimpleMintCompensationConfig,
]

type AvailableCompensation = SimpleMintCompensationConfig

available_contribution_calculators = [
    ContributionCalculatorConfig,
]

type AvailableContributionCalculator = ContributionCalculatorConfig

available_access_controls = [
    BaseAccessControlConfig,
]

type AvailableAccessControl = BaseAccessControlConfig


class SwarmV1FactoryConfig(BaseModel):
    name: str = Field(..., description="The swarm name")
    local_factory_deployment_path: str | None = Field(
        None, description="path to local deployments"
    )

    factory_deployments: dict[int, DeployedContract] = {
        RIZENET_TESTNET_CHAINID: DeployedContract(
            address=Web3.to_checksum_address(
                "0xd66c7c89fb97ea5c06b0b7caf2086df1e82b9e88"
            )
        )
    }

    trainer_selector: AvailableSelector = Field(
        AlwaysSamplesSelectorConfig(),
        description="The trainer selector to use",
    )

    evaluator_selector: AvailableSelector = Field(
        AlwaysSamplesSelectorConfig(),
        description="The evaluator selector to use",
    )
    access_control: AvailableAccessControl = Field(
        description="The access control to use",
    )

    compensation: AvailableCompensation = Field(
        description="The compensation to use",
    )

    contribution_calculator: AvailableContributionCalculator = Field(
        description="The contribution calculator to use",
    )

    training_phase: BaseTrainingPhaseConfig = Field(
        description="The training phase to use",
    )

    evaluation_phase: BaseEvaluationPhaseConfig = Field(
        description="The evaluation phase to use",
    )

    def __init__(self, **data):
        super().__init__(**data)

    def get_contract_name(self) -> str:
        return "SwarmV1Factory"

    def get_factory_deployment(self, chain_id: int) -> DeployedContract:
        if self.local_factory_deployment_path is not None:
            return load_forge_artifact(
                Path(self.local_factory_deployment_path), self.get_contract_name()
            )
        if chain_id in self.factory_deployments:
            return self.factory_deployments[chain_id]
        raise Exception(
            f"Chain ID#{chain_id} is unsupported, provide a local_deployment_path"
        )


abi = load_abi(Path(os.path.dirname(__file__)) / "./factory_abi.json")


class SwarmV1Factory:
    config: SwarmV1FactoryConfig

    def __init__(self, config: SwarmV1FactoryConfig):
        self.config = config

    def get_factory(self, w3: Web3) -> Contract:
        factory_meta = self.config.get_factory_deployment(w3.eth.chain_id)
        factory = w3.eth.contract(abi=abi, address=factory_meta.address_as_bytes())
        return factory

    def deploy(self, deployer: BaseAccount, w3: Web3):
        factory = self.get_factory(w3)
        log(
            INFO,
            "Web3 swarm factory contract address: %s",
            Web3.to_checksum_address(factory.address),
        )

        salt = Web3.keccak(os.urandom(32))
        swarm_address = self.get_swarm_address(salt, w3)

        trainer_selector_params = self.config.trainer_selector.get_selector_params()
        evaluator_selector_params = self.config.evaluator_selector.get_selector_params()
        compensation_params = self.config.compensation.get_compensation_params(
            swarm_address=swarm_address
        )
        contribution_calculator_params = (
            self.config.contribution_calculator.get_calculator_params(
                swarm_address=swarm_address
            )
        )
        access_control_params = self.config.access_control.get_access_control_params(
            swarm_address=swarm_address
        )
        swarm_params = {
            "swarm": {
                "name": self.config.name,
            },
            "trainerSelector": {
                "id": trainer_selector_params.id,
                "initData": trainer_selector_params.init_data,
            },
            "evaluatorSelector": {
                "id": evaluator_selector_params.id,
                "initData": evaluator_selector_params.init_data,
            },
            "compensation": {
                "id": compensation_params.id,
                "initData": compensation_params.init_data,
            },
            "contributionCalculator": {
                "id": contribution_calculator_params.id,
                "initData": contribution_calculator_params.init_data,
            },
            "accessControl": {
                "id": access_control_params.id,
                "initData": access_control_params.init_data,
            },
            "trainingPhaseConfiguration": self.config.training_phase.to_struct(),
            "evaluationPhaseConfiguration": self.config.evaluation_phase.to_struct(),
        }

        tx = factory.functions.createSwarm(salt, swarm_params).build_transaction(
            {
                "from": deployer.address,
                "nonce": w3.eth.get_transaction_count(deployer.address),
            }
        )

        signed_tx = deployer.sign_transaction(cast(TransactionDictType, tx))

        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt["status"] != 0, "Deployment transaction failed or reverted."

        # logs = factory.events.ContractCreated().process_receipt(tx_receipt)

        logs = decode_events_from_tx(
            tx_hash=tx_hash, event=factory.events.ContractCreated, w3=w3
        )
        assert len(logs) == 1, "no events discovered, factory might not be deployed"
        contract_created = logs[0]
        proxy_address = contract_created["args"]["proxyAddress"]

        return DeployedContract(address=proxy_address)

    def get_swarm_address(self, salt: HexBytes, w3: Web3) -> ChecksumAddress:
        factory = self.get_factory(w3)
        return factory.functions.getSwarmAddress(salt).call()
