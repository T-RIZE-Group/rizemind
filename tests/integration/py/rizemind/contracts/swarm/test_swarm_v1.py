import pytest
from eth_account.signers.base import BaseAccount
from rizemind.contracts.access_control.base_access_control.base_access_control import (
    BaseAccessControlConfig,
)
from rizemind.contracts.compensation.simple_mint_compensation.simple_mint_compensation import (
    SimpleMintCompensationConfig,
)
from rizemind.contracts.contribution.contribution_calculator.contribution_calculator import (
    ContributionCalculatorConfig,
)
from rizemind.contracts.swarm.swarm_v1.swarm_v1 import SwarmV1
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import (
    SwarmV1Factory,
    SwarmV1FactoryConfig,
)
from rizemind.contracts.swarm.training.base_training_phase.config import (
    BaseEvaluationPhaseConfig,
    BaseTrainingPhaseConfig,
)

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type SwarmFixture = tuple[SwarmV1, BaseAccount, list[BaseAccount]]


@pytest.fixture(scope="module")
def swarm_deployment(anvil: AnvilContext):
    aggregator = anvil.account_conf.get_account(0)
    trainers = [anvil.account_conf.get_account(i) for i in range(2, 4)]
    evaluators = [anvil.account_conf.get_account(i) for i in range(4, 6)]
    artifact_path = run_script(
        "script/integrations/Deploy.s.sol:DeployAll",
        account=aggregator.address,
        env={"OWNER": aggregator.address},
    )
    factory_config = SwarmV1FactoryConfig(
        name="test",
        local_factory_deployment_path=str(artifact_path),
        access_control=BaseAccessControlConfig(
            aggregator=aggregator.address,
            trainers=[trainer.address for trainer in trainers],
            evaluators=[evaluator.address for evaluator in evaluators],
        ),
        compensation=SimpleMintCompensationConfig(
            token_symbol="tst",
            token_name="test",
            target_rewards=10**18,
        ),
        contribution_calculator=ContributionCalculatorConfig(
            initial_num_samples=1,
        ),
        training_phase=BaseTrainingPhaseConfig(ttl=1000),
        evaluation_phase=BaseEvaluationPhaseConfig(ttl=1000, registration_ttl=1000),
    )
    w3 = anvil.w3conf.get_web3()
    factory = SwarmV1Factory(factory_config)
    swarm_deployment = factory.deploy(aggregator, w3)
    swarm = SwarmV1.from_address(
        account=aggregator, w3=w3, address=swarm_deployment.address
    )
    return swarm, aggregator, trainers


def test_can_train(swarm_deployment: SwarmFixture):
    swarm, aggregator, trainers = swarm_deployment
    # Trainers should be able to train

    assert swarm.can_train(trainers[0].address, 1) is True
    # Aggregator should not be able to train
    assert swarm.can_train(aggregator.address, 1) is False


def test_distribute(swarm_deployment: SwarmFixture, anvil: AnvilContext):
    swarm, aggregator, trainers = swarm_deployment
    trainer_scores = [
        (trainer.address, 0.9 + i * 0.01) for i, trainer in enumerate(trainers)
    ]
    swarm.connect(aggregator)
    tx_hash = swarm.distribute(trainer_scores)
    w3 = anvil.w3conf.get_web3()
    receipt = w3.eth.get_transaction_receipt(tx_hash)
    assert receipt["status"] == 1
