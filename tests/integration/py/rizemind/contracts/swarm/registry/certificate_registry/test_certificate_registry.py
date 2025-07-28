import pytest
from eth_account.signers.base import BaseAccount
from rizemind.contracts.swarm.registry.certificate_registry.certificate_registry import (
    CertificateRegistry,
)

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import deploy_contract

type Deployment = tuple[CertificateRegistry, BaseAccount]


@pytest.fixture(scope="module")
def deploy_certificate_registry(anvil: AnvilContext) -> Deployment:
    owner = anvil.account_conf.get_account(0)
    contract = deploy_contract(
        "test/swarm/registry/CertificateRegistry.t.sol",
        "TestCertificateRegistry",
        account=owner.address,
    )
    w3 = anvil.w3conf.get_web3()
    registry = CertificateRegistry.from_address(
        address=contract.address, w3=w3, account=owner
    )
    registry.connect(owner)
    return registry, owner


def test_set_and_get_certificate(deploy_certificate_registry: Deployment):
    registry, owner = deploy_certificate_registry
    cert_id = "test-cert"
    cert_value = b"certificate-data"
    tx_hash = registry.set_certificate(cert_id, cert_value)
    assert tx_hash is not None

    result = registry.get_certificate(cert_id)
    assert result == cert_value
