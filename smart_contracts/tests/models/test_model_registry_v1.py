import ape
import pytest

@pytest.fixture
def aggregator(accounts):
    return accounts[0]

@pytest.fixture
def trainers(accounts):
    return accounts[1:4]  

@pytest.fixture
def non_trainer(accounts):
    return accounts[4]

@pytest.fixture
def model_contract(project, aggregator, trainers):
    mc = aggregator.deploy(project.ModelRegistryV1)
    mc.initialize("Test", "tst", aggregator, trainers, sender=aggregator)
    return mc


@pytest.fixture
def model_factory(project, aggregator, model_contract):
    return aggregator.deploy(project.ModelRegistryFactory, model_contract)

def test_factory_deploy(project, aggregator, model_factory, trainers):
    tx = model_factory.createModel("hello", "world", aggregator, trainers, sender=aggregator)

    # This should work, but it doesnt :( https://github.com/ApeWorX/ape/pull/1392
    # assert project.ModelRegistryFactory.ContractCreated() in tx.events, "ContractCreated not in events"
    assert tx.events[-1].event_name == "ContractCreated"
