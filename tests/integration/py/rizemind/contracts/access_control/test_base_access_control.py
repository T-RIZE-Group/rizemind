import os

import pytest
from eth_typing import ChecksumAddress
from rizemind.contracts.access_control.access_control_factory import (
    AccessControlFactoryContract,
)
from rizemind.contracts.access_control.base_access_control.base_access_control import (
    BaseAccessControl,
    BaseAccessControlConfig,
)
from rizemind.contracts.local_deployment import load_forge_artifact

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type Deployment = tuple[
    BaseAccessControl, ChecksumAddress, list[ChecksumAddress], list[ChecksumAddress]
]


@pytest.fixture(scope="module")
def deploy_access_control(anvil: AnvilContext):
    aggregator = anvil.account_conf.get_account(0)
    trainers = [anvil.account_conf.get_account(i).address for i in range(1, 4)]
    evaluators = [anvil.account_conf.get_account(i).address for i in range(4, 7)]
    artifact_path = run_script(
        "script/integrations/Deploy.s.sol:DeployAll",
        account=aggregator.address,
        env={"OWNER": aggregator.address},
    )
    contract = load_forge_artifact(artifact_path, "AccessControlFactory")
    w3 = anvil.w3conf.get_web3()
    factory = AccessControlFactoryContract.from_address(
        address=contract.address, w3=w3, account=aggregator
    )

    config = BaseAccessControlConfig(
        aggregator=aggregator.address,
        trainers=trainers,
        evaluators=evaluators,
    )
    deploy_tx = factory.create_access_control(
        access_control_id=config.get_access_control_id(),
        salt=w3.keccak(text="test-base-access-control"),
        init_data=config.get_init_data(),
    )
    ac = factory.get_deployed_access_control(deploy_tx)
    fl_access = BaseAccessControl.from_address(address=ac, w3=w3)
    fl_access.connect(anvil.account_conf.get_account(0))
    return fl_access, aggregator.address, trainers, evaluators


def test_is_trainer_and_aggregator(deploy_access_control: Deployment):
    fl_access, aggregator, trainers, evaluators = deploy_access_control
    # aggregator should not be trainer
    assert not fl_access.is_trainer(aggregator)
    # trainers[0] should be trainer
    assert fl_access.is_trainer(trainers[0])
    # aggregator should be aggregator
    assert fl_access.is_aggregator(aggregator)
    # trainers[0] should not be aggregator
    assert not fl_access.is_aggregator(trainers[0])
    # evaluators[0] should be evaluator
    assert fl_access.is_evaluator(evaluators[0])
    # evaluators[0] should not be trainer
    assert not fl_access.is_trainer(evaluators[0])
    # evaluators[0] should not be aggregator
    assert not fl_access.is_aggregator(evaluators[0])
