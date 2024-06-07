import ape
import pytest

@pytest.fixture
def account(accounts):
    return accounts[0]

@pytest.fixture
def helloword(account, project):
    return account.deploy(project.HelloWorld)

def test_increment(helloword, account):
    helloword.increment(sender=account)
    print(helloword.getCounter(sender=account))
    assert helloword.getCounter() == 1