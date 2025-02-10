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

def test_factory_deploy(aggregator, model_factory, trainers):
    tx = model_factory.createModel("hello", "world", aggregator, trainers, sender=aggregator)

    assert model_factory.ContractCreated() in tx.events, "ContractCreated not in events"

