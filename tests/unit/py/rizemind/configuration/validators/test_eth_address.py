import pytest
from rizemind.configuration.validators.eth_address import _validate_eth_address

DEADBEEF_RAW = "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
DEADBEEF_CHECKSUM = "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (DEADBEEF_RAW, DEADBEEF_CHECKSUM),  # lower-case to checksummed
        (DEADBEEF_CHECKSUM, DEADBEEF_CHECKSUM),  # already checksummed
    ],
)
def test_validate_eth_address_success(raw, expected):
    assert _validate_eth_address(raw) == expected


@pytest.mark.parametrize(
    "bad_value, exc_type",
    [
        (123, TypeError),  # not a string
        ("0x123456", ValueError),  # wrong length
    ],
)
def test_validate_eth_address_failure(bad_value, exc_type):
    with pytest.raises(exc_type):
        _validate_eth_address(bad_value)
