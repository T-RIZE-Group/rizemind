import pytest
from eth_typing import ChecksumAddress
from rizemind.authentication.authenticated_client_properties import (
    AUTHENTICATED_CLIENT_PROPERTIES_PREFIX,
    AuthenticatedClientProperties,
)
from web3 import Web3


class DummyClientProxy:
    """A dummy implementation of ClientProxy for testing purposes."""

    def __init__(self, cid: str):
        self.cid = cid
        self.properties = {}


class TestAuthenticatedClientProperties:
    """Test suite for AuthenticatedClientProperties class."""

    @pytest.fixture
    def sample_trainer_address(self) -> ChecksumAddress:
        """Create a sample trainer address for testing."""
        return Web3.to_checksum_address("0x1234567890123456789012345678901234567890")

    @pytest.fixture
    def authenticated_client_properties(
        self, sample_trainer_address
    ) -> AuthenticatedClientProperties:
        """Create an AuthenticatedClientProperties instance for testing."""
        return AuthenticatedClientProperties(trainer_address=sample_trainer_address)

    @pytest.fixture
    def mock_client_proxy(self) -> DummyClientProxy:
        """Create a mock ClientProxy for testing."""
        mock_client = DummyClientProxy(cid="test_client")
        return mock_client

    def test_init_with_valid_address(self, sample_trainer_address):
        """Test that AuthenticatedClientProperties can be initialized with a valid address."""
        props = AuthenticatedClientProperties(trainer_address=sample_trainer_address)
        assert props.trainer_address == sample_trainer_address
        assert isinstance(props.trainer_address, str)
        assert Web3.is_address(props.trainer_address)

    def test_init_with_different_address(self):
        """Test that AuthenticatedClientProperties can be initialized with a different address."""
        address = Web3.to_checksum_address("0x0987654321098765432109876543210987654321")
        props = AuthenticatedClientProperties(trainer_address=address)
        assert props.trainer_address == address
        assert Web3.is_address(props.trainer_address)

    def test_tag_client_updates_client_properties(
        self, authenticated_client_properties, mock_client_proxy
    ):
        """Test that tag_client properly updates the client properties."""

        authenticated_client_properties.tag_client(mock_client_proxy)

        # Verify properties were added with correct prefix
        expected_key = f"{AUTHENTICATED_CLIENT_PROPERTIES_PREFIX}.trainer_address"
        assert expected_key in mock_client_proxy.properties
        assert (
            mock_client_proxy.properties[expected_key]
            == authenticated_client_properties.trainer_address
        )

    def test_from_client_returns_correct_instance(
        self, sample_trainer_address, mock_client_proxy
    ):
        """Test that from_client returns the correct AuthenticatedClientProperties instance."""
        # First tag the client with properties
        props = AuthenticatedClientProperties(trainer_address=sample_trainer_address)
        props.tag_client(mock_client_proxy)

        # Now extract properties back
        result = AuthenticatedClientProperties.from_client(mock_client_proxy)

        # Verify the result is correct
        assert isinstance(result, AuthenticatedClientProperties)
        assert result.trainer_address == sample_trainer_address
        assert Web3.is_address(result.trainer_address)

    def test_tag_client_preserves_existing_properties(
        self, authenticated_client_properties, mock_client_proxy
    ):
        """Test that tag_client preserves existing client properties."""
        # Set existing properties
        existing_properties = {"existing.key": "existing.value"}
        mock_client_proxy.properties = existing_properties.copy()

        authenticated_client_properties.tag_client(mock_client_proxy)

        # Verify both existing and new properties are present
        assert "existing.key" in mock_client_proxy.properties
        assert mock_client_proxy.properties["existing.key"] == "existing.value"

        expected_key = f"{AUTHENTICATED_CLIENT_PROPERTIES_PREFIX}.trainer_address"
        assert expected_key in mock_client_proxy.properties
        assert (
            mock_client_proxy.properties[expected_key]
            == authenticated_client_properties.trainer_address
        )

    def test_round_trip_properties(self, sample_trainer_address, mock_client_proxy):
        """Test that properties can be written and read back correctly."""
        # Create and tag client
        original_props = AuthenticatedClientProperties(
            trainer_address=sample_trainer_address
        )
        original_props.tag_client(mock_client_proxy)

        # Read back properties
        extracted_props = AuthenticatedClientProperties.from_client(mock_client_proxy)

        # Verify round trip
        assert extracted_props.trainer_address == original_props.trainer_address
        assert extracted_props.model_dump() == original_props.model_dump()
