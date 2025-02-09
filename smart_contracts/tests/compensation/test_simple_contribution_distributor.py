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


max_rewards = 3 * 10 ** 18

@pytest.fixture
def token(project, aggregator):
    token = aggregator.deploy(project.TESTSimpleContributionDistributor)
    token.initialize("Test", "tst", max_rewards, sender=aggregator)
    return token


def test_distribute(token, aggregator, trainers):
    token.distribute(trainers, [10**6, 2*10**3, 50], sender=aggregator)
    assert token.balanceOf(trainers[0]) == max_rewards, "should have received max rewards"
    assert token.balanceOf(trainers[1]) == (2 * max_rewards / 10**3), "should not receive max rewards"
    assert token.balanceOf(trainers[2]) == (50 * max_rewards / 10**6), "should not receive max rewards"