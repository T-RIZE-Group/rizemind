import pytest
from ape.exceptions import ContractLogicError


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
    tx = model_factory.createModel(
        "hello", "world", aggregator, trainers, sender=aggregator
    )

    assert model_factory.ContractCreated() in tx.events, "ContractCreated not in events"


def test_update_implementation(project, aggregator, model_factory, trainers):
    new_model = aggregator.deploy(project.ModelRegistryV1)
    new_model.initialize("Test2", "tst", aggregator, trainers, sender=aggregator)
    model_factory.updateImplementation(new_model, sender=aggregator)

    model_factory.createModel("hello", "world", aggregator, trainers, sender=aggregator)


def test_update_implementation_protected(model_factory, trainers):
    with pytest.raises(ContractLogicError):
        model_factory.updateImplementation(trainers[0], sender=trainers[0])
