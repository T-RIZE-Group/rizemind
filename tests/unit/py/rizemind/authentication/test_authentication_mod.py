import pytest
from eth_account.signers.base import BaseAccount
from flwr.common import (
    ConfigRecord,
    Context,
    FitIns,
    GetPropertiesIns,
    Message,
    MessageType,
    Parameters,
    RecordDict,
)
from flwr.common.constant import MessageTypeLegacy
from flwr.common.recorddict_compat import (
    fitins_to_recorddict,
    getpropertiesins_to_recorddict,
    recorddict_to_getpropertiesres,
)
from rizemind.authentication import (
    authentication_mod,
)
from rizemind.authentication.config import ACCOUNT_CONFIG_STATE_KEY, AccountConfig
from rizemind.authentication.mod import (
    NoAccountAuthenticationModError,
    WrongSwarmAuthenticationModError,
)
from rizemind.authentication.signatures.auth import recover_auth_signer
from rizemind.authentication.train_auth import (
    TrainAuthInsConfig,
    parse_train_auth_res,
    prepare_train_auth_ins,
)
from rizemind.contracts.erc.erc5267.typings import EIP712DomainMinimal
from rizemind.swarm.config import SWARM_CONFIG_STATE_KEY, SwarmConfig
from web3 import Web3

"""
tests
- should return reply if not a get property
- should return reply if parsing fails
- should raise if no account
- should raise if no swarm config
- should raise if wrong swarm domain
- should return a message with properly signed auth message
"""


class TestAuthenticationMod:
    """Test suite for authentication_mod function."""

    @pytest.fixture
    def filled_context(
        self,
        context: Context,
        account_config: AccountConfig,
        real_swarm_config: SwarmConfig,
    ) -> Context:
        """Create a context with all required configurations properly set up."""

        # Add account config
        context.state.config_records[ACCOUNT_CONFIG_STATE_KEY] = (
            account_config.to_config_record()
        )

        # Add swarm config
        context.state.config_records[SWARM_CONFIG_STATE_KEY] = (
            real_swarm_config.to_config_record()
        )

        return context

    @pytest.fixture
    def call_next(self):
        """Create a call_next function that returns a proper Message."""

        def call_next(msg: Message, ctxt: Context) -> Message:
            content = RecordDict()
            content.config_records["rizemind.mocked"] = ConfigRecord({"reply": "data"})
            reply = Message(
                content=content,
                reply_to=msg,
            )
            return reply

        return call_next

    @pytest.fixture
    def sample_train_auth_ins(
        self, minimal_domain: EIP712DomainMinimal
    ) -> TrainAuthInsConfig:
        """Create sample train auth instructions."""
        return TrainAuthInsConfig(
            domain=minimal_domain, round_id=1, nonce=b"test_nonce_32_bytes_long_data_"
        )

    @pytest.fixture
    def real_swarm_config(self, minimal_domain: EIP712DomainMinimal) -> SwarmConfig:
        """Create a real SwarmConfig instance."""
        return SwarmConfig(address=minimal_domain.verifyingContract)

    @pytest.fixture
    def wrong_swarm_config(self) -> SwarmConfig:
        """Create a SwarmConfig with different address."""
        return SwarmConfig(
            address=Web3.to_checksum_address(
                "0x0987654321098765432109876543210987654321"
            )
        )

    @pytest.fixture
    def get_properties_message(
        self, sample_train_auth_ins: TrainAuthInsConfig
    ) -> Message:
        """Create a real GET_PROPERTIES message with train auth data."""
        # Create the train auth instructions
        train_auth_ins = prepare_train_auth_ins(
            round_id=sample_train_auth_ins.round_id,
            nonce=sample_train_auth_ins.nonce,
            domain=sample_train_auth_ins.domain,
        )

        # Convert to recorddict format
        recorddict = getpropertiesins_to_recorddict(train_auth_ins)

        # Create the message
        msg = Message(
            content=recorddict,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
            dst_node_id=1,
        )
        return msg

    @pytest.fixture
    def non_get_properties_message(self) -> Message:
        """Create a non-GET_PROPERTIES message."""
        ins = FitIns(
            parameters=Parameters(tensors=[], tensor_type="empty"), config={"test": 1}
        )
        msg = Message(
            content=fitins_to_recorddict(ins, True),
            message_type=MessageType.TRAIN,
            dst_node_id=1,
        )
        return msg

    def test_authentication_mod_recoverrable_signature(
        self,
        filled_context: Context,
        call_next,
        get_properties_message: Message,
        account: BaseAccount,
        minimal_domain: EIP712DomainMinimal,
        sample_train_auth_ins,
    ) -> None:
        """Test successful authentication flow."""
        # Act
        result = authentication_mod(get_properties_message, filled_context, call_next)

        # Assert
        assert isinstance(result, Message)
        get_properties_res = recorddict_to_getpropertiesres(result.content)
        train_auth_res = parse_train_auth_res(get_properties_res)

        signer = recover_auth_signer(
            round=sample_train_auth_ins.round_id,
            nonce=sample_train_auth_ins.nonce,
            domain=minimal_domain,
            signature=train_auth_res.signature,
        )
        assert signer == account.address

    def test_authentication_mod_non_get_properties_message(
        self, context: Context, call_next, non_get_properties_message: Message
    ) -> None:
        """Test that non-GET_PROPERTIES messages pass through unchanged."""
        # Act
        result = authentication_mod(non_get_properties_message, context, call_next)

        assert result.metadata.message_type == MessageType.TRAIN

    def test_authentication_mod_no_account_config(
        self, context: Context, call_next, get_properties_message: Message
    ) -> None:
        """Test that NoAccountAuthenticationModError is raised when no account config is found."""
        # Act & Assert
        with pytest.raises(NoAccountAuthenticationModError) as exc_info:
            authentication_mod(get_properties_message, context, call_next)

        assert exc_info.value.code == "no_account_config"

    def test_authentication_mod_wrong_swarm_domain(
        self,
        context: Context,
        call_next,
        account_config: AccountConfig,
        get_properties_message: Message,
        wrong_swarm_config: SwarmConfig,
    ) -> None:
        """Test that WrongSwarmAuthenticationModError is raised when swarm domains don't match."""
        # Arrange - Add account config to context
        context.state.config_records[ACCOUNT_CONFIG_STATE_KEY] = (
            account_config.to_config_record()
        )

        # Add wrong swarm config to context
        context.state.config_records[SWARM_CONFIG_STATE_KEY] = (
            wrong_swarm_config.to_config_record()
        )

        # Act & Assert
        with pytest.raises(WrongSwarmAuthenticationModError) as exc_info:
            authentication_mod(get_properties_message, context, call_next)

        assert exc_info.value.code == "wrong_swarm_domain"

    def test_authentication_mod_parse_exception_handled(
        self, context: Context, call_next, account_config: AccountConfig
    ) -> None:
        """Test that ParseException is handled gracefully and original reply is returned."""
        # Arrange - Create a message with invalid content that will cause ParseException

        msg = Message(
            content=getpropertiesins_to_recorddict(GetPropertiesIns({})),
            message_type=MessageTypeLegacy.GET_PROPERTIES,
            dst_node_id=1,
        )

        # Add account config to context
        context.state.config_records[ACCOUNT_CONFIG_STATE_KEY] = (
            account_config.to_config_record()
        )

        # Act
        result = authentication_mod(msg, context, call_next)

        # Assert - Should return the original reply since ParseException was caught
        assert result.content.config_records["rizemind.mocked"]["reply"] == "data"

    def test_authentication_mod_with_fallback_address(
        self,
        context: Context,
        call_next,
        account_config: AccountConfig,
        sample_train_auth_ins: TrainAuthInsConfig,
        account: BaseAccount,
        minimal_domain: EIP712DomainMinimal,
    ) -> None:
        """Test authentication with fallback address from train auth ins."""
        # Arrange - Add account config to context
        context.state.config_records[ACCOUNT_CONFIG_STATE_KEY] = (
            account_config.to_config_record()
        )

        # Create message with train auth data
        train_auth_ins = prepare_train_auth_ins(
            round_id=sample_train_auth_ins.round_id,
            nonce=sample_train_auth_ins.nonce,
            domain=sample_train_auth_ins.domain,
        )

        recorddict = getpropertiesins_to_recorddict(train_auth_ins)
        msg = Message(
            content=recorddict,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
            dst_node_id=1,
        )

        # Act
        result = authentication_mod(msg, context, call_next)

        # Assert
        assert isinstance(result, Message)
        get_properties_res = recorddict_to_getpropertiesres(result.content)
        train_auth_res = parse_train_auth_res(get_properties_res)

        signer = recover_auth_signer(
            round=sample_train_auth_ins.round_id,
            nonce=sample_train_auth_ins.nonce,
            domain=minimal_domain,
            signature=train_auth_res.signature,
        )
        assert signer == account.address
