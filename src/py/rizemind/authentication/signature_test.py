import pytest
from eth_account import Account
from eth_account.datastructures import SignedMessage
from .signature import (
    sign_parameters_model,
    prepare_eip712_message,
    hash_parameters,
    Parameters,
)


@pytest.fixture
def eth_account():
    # Create a test Ethereum account
    return Account.create()


def test_sign_parameters_model(eth_account):
    chain_id = 1
    contract_address = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"
    app_name = "TestApp"
    round_number = 1
    model = Parameters(tensors=[bytes(0x87231), bytes(0x5423)], tensor_type="float32")

    signed_message = sign_parameters_model(
        eth_account, "1.0.0", model, chain_id, contract_address, app_name, round_number
    )
    print(type(signed_message))
    assert isinstance(signed_message, SignedMessage), "should return a SignedMessage"

    message = prepare_eip712_message(
        chain_id,
        "1.0.0",
        contract_address,
        app_name,
        round_number,
        hash_parameters(model),
    )
    address = Account.recover_message(
        message, [signed_message.v, signed_message.r, signed_message.s]
    )
    assert address == eth_account.address, "recovered address doesn't match"
