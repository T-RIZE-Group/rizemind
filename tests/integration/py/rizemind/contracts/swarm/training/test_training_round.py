from collections.abc import Sequence

import pytest
from eth_account.signers.base import BaseAccount
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import (
    SwarmV1FactoryConfig,
)
from rizemind.contracts.swarm.training.round_training import RoundTraining
from rizemind.swarm.config import SwarmConfig
from rizemind.swarm.swarm import Swarm

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type SwarmFixture = tuple[RoundTraining, Swarm, BaseAccount, Sequence[BaseAccount]]


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
    training = RoundTraining.from_address(
        w3=w3, address=swarm.training.address, account=aggregator
    )
    return training, swarm, aggregator, trainers


def test_model_meta_get_round_at(swarm_deployment, anvil):
    training, swarm, aggregator, trainers = swarm_deployment
    w3 = anvil.w3conf.get_web3()
    current_block: int = w3.eth.block_number
    summary = training.get_round_at(current_block)

    assert summary.round_id == 1, "First round should be #1"
    assert summary.finished is False, "No rounds have finished yet"
    assert summary.metrics is None, "No metrics until a round finishes"

    ## finish round 1

    round_1_model_score: float = 0.9
    round_1_n_trainers: int = 2
    round_1_total_contribution: int = 3

    tx_hash = swarm.next_round(
        1,
        round_1_n_trainers,
        round_1_model_score,
        round_1_total_contribution,
    )

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    round_1_block: int = receipt["blockNumber"]

    summary = training.get_round_at(round_1_block)

    assert summary.finished is True
    assert summary.round_id == 1
    assert summary.metrics is not None
    assert summary.metrics.n_trainers == round_1_n_trainers
    assert summary.metrics.model_score == pytest.approx(round_1_model_score)
    assert summary.metrics.total_contributions == round_1_total_contribution

    later_block: int = round_1_block + 1
    future_summary = training.get_round_at(later_block)

    assert future_summary.finished is False
    assert future_summary.metrics is None
    assert future_summary.round_id == 2

    round_2_score: float = 0.95
    round_2_trainers: int = 3
    round_2_total_contrib: int = 5

    tx_hash = swarm.next_round(
        2, round_2_trainers, round_2_score, round_2_total_contrib
    )
    # ---- exact-block query  → finished == True for round 2
    r2_summary = training.get_round_at(later_block)
    assert r2_summary.finished is True
    assert r2_summary.round_id == 2
    assert r2_summary.metrics is not None
    assert r2_summary.metrics.n_trainers == round_2_trainers
    assert r2_summary.metrics.model_score == pytest.approx(round_2_score)
    assert r2_summary.metrics.total_contributions == round_2_total_contrib
