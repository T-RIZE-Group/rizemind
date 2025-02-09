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
def access_control(project, aggregator, trainers):
    ac = aggregator.deploy(project.InitializableFLAccessControl)
    ac.initialize(aggregator, trainers, sender=aggregator)
    return ac


def test_is_trainer(access_control, aggregator, trainers, non_trainer):
    assert access_control.isTrainer(aggregator) is False, "aggregator is trainer"
    assert access_control.isTrainer(trainers[0]) is True, "trainer role not assigned"
    assert access_control.isTrainer(non_trainer) is False, "non-trainer is trainer"

def test_is_aggregator(access_control, aggregator, trainers, non_trainer):
    assert access_control.isAggregator(aggregator) is True, "aggregator role not assigned"
    assert access_control.isAggregator(trainers[0]) is False, "trainers is aggregator"
    assert access_control.isAggregator(non_trainer) is False, "non-trainer is aggregator"