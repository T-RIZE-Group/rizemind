import pytest
from rizemind.contracts.access_control.base_access_control.base_access_control import (
    BaseAccessControlConfig,
)
from rizemind.contracts.compensation.simple_mint_compensation.simple_mint_compensation import (
    SimpleMintCompensationConfig,
)
from rizemind.contracts.contribution.contribution_calculator.contribution_calculator import (
    ContributionCalculatorConfig,
)
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


@pytest.fixture(scope="module")
def factory_config(anvil: AnvilContext):
    deployer = anvil.account_conf.get_account(0)
    trainers = [anvil.account_conf.get_account(i) for i in range(2, 4)]
    evaluators = [anvil.account_conf.get_account(i) for i in range(4, 6)]
    artifact_path = run_script(
        "script/integrations/Deploy.s.sol:DeployAll",
        account=deployer.address,
        env={"OWNER": deployer.address},
    )
    factory_config = SwarmV1FactoryConfig(
        name="test",
        local_factory_deployment_path=str(artifact_path),
        access_control=BaseAccessControlConfig(
            aggregator=deployer.address,
            trainers=[trainer.address for trainer in trainers],
            evaluators=[evaluator.address for evaluator in evaluators],
        ),
        compensation=SimpleMintCompensationConfig(
            token_symbol="tst",
            token_name="test",
            target_rewards=10**18,
            initial_admin=deployer.address,
            minter=deployer.address,
        ),
        contribution_calculator=ContributionCalculatorConfig(
            initial_num_samples=1,
        ),
        training_phase=BaseTrainingPhaseConfig(ttl=1000),
        evaluation_phase=BaseEvaluationPhaseConfig(ttl=1000, registration_ttl=1000),
    )

    return factory_config


def test_deploy(factory_config: SwarmV1FactoryConfig, anvil: AnvilContext):
    factory = SwarmV1Factory(factory_config)
    deployer = anvil.account_conf.get_account(1)
    swarm = factory.deploy(deployer, anvil.w3conf.get_web3())
    assert swarm.address is not None
