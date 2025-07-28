import pytest
from eth_account.signers.base import BaseAccount
from rizemind.contracts.swarm.swarm_v1.swarm_v1 import SwarmV1
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import (
    SwarmV1Factory,
    SwarmV1FactoryConfig,
)

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type SwarmFixture = tuple[SwarmV1, BaseAccount, list[BaseAccount]]


@pytest.fixture(scope="module")
def swarm_deployment(anvil: AnvilContext):
    deployer = anvil.account_conf.get_account(0)
    artifact_path = run_script(
        "script/deployments/SwarmV1Factory.s.sol",
        account=deployer.address,
    )
    factory_config = SwarmV1FactoryConfig(
        name="test", local_factory_deployment_path=str(artifact_path)
    )
    factory = SwarmV1Factory(factory_config)
    aggregator = anvil.account_conf.get_account(1)
    trainers = [anvil.account_conf.get_account(i) for i in range(2, 4)]
    w3 = anvil.w3conf.get_web3()
    swarm_deployment = factory.deploy(
        aggregator, [trainer.address for trainer in trainers], w3
    )
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
