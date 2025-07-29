import pytest
from rizemind.contracts.access_control.fl_access_control.FlAccessControl import (
    FlAccessControl,
)

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import deploy_contract

type Deployment = tuple[FlAccessControl, str, list[str]]


@pytest.fixture(scope="module")
def deploy_access_control(anvil: AnvilContext):
    aggregator = anvil.account_conf.get_account(0).address
    trainers = [anvil.account_conf.get_account(i).address for i in range(1, 4)]
    contract = deploy_contract(
        "src/access/FLAccessControl.sol",
        "InitializableFLAccessControl",
        account=aggregator,
    )
    w3 = anvil.w3conf.get_web3()
    fl_access = FlAccessControl.from_address(address=contract.address, w3=w3)
    fl_access.connect(anvil.account_conf.get_account(0))
    fl_access.initialize(aggregator=aggregator, trainers=trainers)
    return fl_access, aggregator, trainers


def test_is_trainer_and_aggregator(deploy_access_control: Deployment):
    fl_access, aggregator, trainers = deploy_access_control
    # aggregator should not be trainer
    assert not fl_access.is_trainer(aggregator)
    # trainers[0] should be trainer
    assert fl_access.is_trainer(trainers[0])
    # aggregator should be aggregator
    assert fl_access.is_aggregator(aggregator)
    # trainers[0] should not be aggregator
    assert not fl_access.is_aggregator(trainers[0])
