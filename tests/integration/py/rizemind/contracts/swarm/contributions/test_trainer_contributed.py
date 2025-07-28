from collections.abc import Sequence

import pytest
from eth_account.signers.base import BaseAccount
from rizemind.contracts.swarm.contributions.trainers_contributed import (
    TrainersContributedEventHelper,
)
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import (
    SwarmV1FactoryConfig,
)
from rizemind.swarm.config import SwarmConfig
from rizemind.swarm.swarm import Swarm

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type SwarmFixture = tuple[
    TrainersContributedEventHelper, Swarm, BaseAccount, Sequence[BaseAccount]
]


@pytest.fixture(scope="module")
def swarm_deployment(anvil: AnvilContext) -> SwarmFixture:
    deployer = anvil.account_conf.get_account(0)
    artifact_path = run_script(
        "script/deployments/SwarmV1Factory.s.sol",
        account=deployer.address,
    )
    factory_config = SwarmV1FactoryConfig(
        name="test", local_factory_deployment_path=str(artifact_path)
    )
    swarm_config = SwarmConfig(factory_v1=factory_config, address=None)
    aggregator = anvil.account_conf.get_account(1)
    trainers = [anvil.account_conf.get_account(i) for i in range(2, 6)]
    w3 = anvil.w3conf.get_web3()
    swarm = swarm_config.get_or_deploy(
        deployer=aggregator,
        trainers=[trainer.address for trainer in trainers],
        w3=w3,
    )
    contribution = TrainersContributedEventHelper.from_address(
        w3=w3, address=swarm.contribution.address
    )
    return contribution, swarm, aggregator, trainers


def test_trainers_contributed_event_helper_get_latest_contribution(
    swarm_deployment: SwarmFixture,
):
    contribution, swarm, aggregator, trainers = swarm_deployment
    trainer1 = trainers[0]
    trainer2 = trainers[1]
    trainer3 = trainers[2]
    swarm.distribute([(trainer1.address, 1.0), (trainer2.address, 2.0)])
    round_1 = {"model_score": 0.9, "n_trainers": 2, "total_contribution": 3}
    swarm.next_round(
        1, round_1["n_trainers"], round_1["n_trainers"], round_1["total_contribution"]
    )
    swarm.distribute([(trainer2.address, 3.0), (trainer3.address, 4.0)])

    trainer1_latest_contribution = contribution.get_latest_contribution(
        trainer1.address
    )
    assert trainer1_latest_contribution == 1.0

    trainer2_latest_contribution = contribution.get_latest_contribution(
        trainer2.address
    )
    assert trainer2_latest_contribution == 3.0

    trainer_3_latest_contribution = contribution.get_latest_contribution(
        trainer3.address
    )
    assert trainer_3_latest_contribution == 4.0

    trainer4 = trainers[3]
    trainer4_latest_contribution = contribution.get_latest_contribution(
        trainer4.address
    )
    assert trainer4_latest_contribution is None
