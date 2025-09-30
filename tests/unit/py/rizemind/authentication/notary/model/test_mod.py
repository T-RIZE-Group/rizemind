from unittest.mock import Mock, patch

import pytest
from eth_account.signers.base import BaseAccount
from flwr.common import (
    Code,
    Context,
    FitIns,
    FitRes,
    Message,
    MessageType,
    Parameters,
    Status,
)
from flwr.common.recorddict_compat import (
    fitins_to_recorddict,
    fitres_to_recorddict,
    recorddict_to_fitres,
)
from rizemind.authentication import AccountConfig
from rizemind.authentication.config import ACCOUNT_CONFIG_STATE_KEY
from rizemind.authentication.notary.model import (
    model_notary_mod,
    parse_model_notary_config,
    prepare_model_notary_config,
    recover_model_signer,
    sign_parameters_model,
)
from rizemind.contracts.erc.erc5267.typings import EIP712DomainMinimal
from rizemind.exception import ParseException, RizemindException
from rizemind.swarm.config import SWARM_CONFIG_STATE_KEY, SwarmConfig
from rizemind.swarm.swarm import Swarm
from rizemind.web3 import Web3Config
from web3 import Web3


class TestModelNotaryMod:
    """Test suite for model_notary_mod function."""

    @pytest.fixture
    def filled_context(
        self,
        context: Context,
        account_config: AccountConfig,
        real_swarm_config: SwarmConfig,
        web3_config: Web3Config,
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

        # Add web3 config
        context.state.config_records["rizemind.web3"] = web3_config.to_config_record()

        return context

    @pytest.fixture
    def call_next(self):
        """Create a call_next function that returns a proper Message."""

        def call_next(msg: Message, ctxt: Context) -> Message:
            # Create a mock FitRes response
            fit_res = FitRes(
                parameters=Parameters(
                    tensors=[b"test_parameters"], tensor_type="bytes"
                ),
                num_examples=10,
                metrics={"accuracy": 0.95},
                status=Status(code=Code.OK, message="OK"),
            )
            content = fitres_to_recorddict(fit_res, False)
            reply = Message(
                content=content,
                reply_to=msg,
            )
            return reply

        return call_next

    @pytest.fixture
    def real_swarm_config(self, minimal_domain: EIP712DomainMinimal) -> SwarmConfig:
        """Create a real SwarmConfig instance."""
        return SwarmConfig(address=minimal_domain.verifyingContract)

    @pytest.fixture
    def train_message(
        self,
        minimal_domain: EIP712DomainMinimal,
        test_round_id: int,
        test_model_hash: bytes,
        account: BaseAccount,
    ) -> Message:
        """Create a TRAIN message with model notary config."""
        parameters = Parameters(tensors=[b"test_model"], tensor_type="bytes")
        signature = sign_parameters_model(
            account=account,
            domain=minimal_domain,
            round=test_round_id,
            parameters=parameters,
        )
        notary_config = prepare_model_notary_config(
            round_id=test_round_id,
            domain=minimal_domain,
            signature=signature,
            model_hash=test_model_hash,
        )

        # Create FitIns with the notary config
        fit_ins = FitIns(
            parameters=parameters,
            config=notary_config,
        )

        # Convert to recorddict format
        recorddict = fitins_to_recorddict(fit_ins, True)

        # Create the message
        msg = Message(
            content=recorddict,
            message_type=MessageType.TRAIN,
            dst_node_id=1,
        )
        return msg

    @pytest.fixture
    def non_train_message(self) -> Message:
        """Create a non-TRAIN message."""
        ins = FitIns(
            parameters=Parameters(tensors=[], tensor_type="empty"), config={"test": 1}
        )
        msg = Message(
            content=fitins_to_recorddict(ins, True),
            message_type=MessageType.EVALUATE,
            dst_node_id=1,
        )
        return msg

    @pytest.fixture
    def test_round_id(self) -> int:
        """Create a test round ID."""
        return 1

    @pytest.fixture
    def test_model_hash(self) -> bytes:
        """Create a test model hash."""
        return Web3.keccak(text="test-model-hash")

    def test_model_notary_mod_success_path(
        self,
        filled_context: Context,
        call_next,
        train_message: Message,
        account: BaseAccount,
        minimal_domain: EIP712DomainMinimal,
        caplog,
    ) -> None:
        """Test the complete success path of model_notary_mod function."""
        swarm_mock = Mock(spec=Swarm)
        mock_config = Mock(spec=SwarmConfig)
        mock_config.get.return_value = swarm_mock
        swarm_mock.is_aggregator.return_value = True
        swarm_mock.get_eip712_domain.return_value = minimal_domain
        with patch("rizemind.swarm.config.SwarmConfig", return_value=mock_config):
            result = model_notary_mod(train_message, filled_context, call_next)
            assert len(caplog.records) == 0, (
                "No logs should be emitted during successful execution"
            )
            fit_res = recorddict_to_fitres(result.content, False)
            signature_config = parse_model_notary_config(fit_res.metrics)
            signer = recover_model_signer(
                signature=signature_config.signature,
                domain=signature_config.domain,
                round=signature_config.round_id,
                model=fit_res.parameters,
            )
            assert signer == account.address, "Signer should match the account address"

    def test_model_notary_mod_parse_exception_logs(
        self,
        filled_context: Context,
        call_next,
        minimal_domain: EIP712DomainMinimal,
        test_round_id: int,
        caplog,
    ) -> None:
        """Test that logs are emitted when ParseException occurs."""
        parameters = Parameters(tensors=[b"test_model"], tensor_type="bytes")
        # Create FitIns with an invalid config
        fit_ins = FitIns(
            parameters=parameters,
            config={
                "round_id": test_round_id,
                # Missing domain and signature fields to trigger ParseException
            },
        )

        # Convert to recorddict format
        recorddict = fitins_to_recorddict(fit_ins, True)

        # Create the message
        invalid_message = Message(
            content=recorddict,
            message_type=MessageType.TRAIN,
            dst_node_id=1,
        )

        # Mock swarm methods
        swarm_mock = Mock(spec=Swarm)
        mock_config = Mock(spec=SwarmConfig)
        mock_config.get.return_value = swarm_mock
        swarm_mock.is_aggregator.return_value = True
        swarm_mock.get_eip712_domain.return_value = minimal_domain

        with patch("rizemind.swarm.config.SwarmConfig", return_value=mock_config):
            # Act - this should trigger a ParseException
            result = model_notary_mod(invalid_message, filled_context, call_next)

            # Assert that logs were emitted due to ParseException
            assert len(caplog.records) > 0, (
                "Logs should be emitted when ParseException occurs"
            )

            # Verify the log contains error information
            log_records = [
                record for record in caplog.records if record.levelname in ["WARNING"]
            ]
            assert len(log_records) == 1, "Should contain warning level logs"

            assert isinstance(result, Message), (
                "Should return a Message even with ParseException"
            )

    def test_model_notary_mod_not_aggregator_exception(
        self,
        filled_context: Context,
        call_next,
        train_message: Message,
        account: BaseAccount,
        minimal_domain: EIP712DomainMinimal,
        caplog,
    ) -> None:
        """Test that an exception is raised when the signer is not an aggregator."""
        # Mock swarm methods to simulate non-aggregator
        swarm_mock = Mock(spec=Swarm)
        mock_config = Mock(spec=SwarmConfig)
        mock_config.get.return_value = swarm_mock
        swarm_mock.is_aggregator.return_value = False  # Not an aggregator
        swarm_mock.get_eip712_domain.return_value = minimal_domain

        with patch("rizemind.swarm.config.SwarmConfig", return_value=mock_config):
            # Act & Assert - should raise an exception
            with pytest.raises(RizemindException) as exc_info:
                model_notary_mod(train_message, filled_context, call_next)

            # Verify the exception message indicates non-aggregator status
            assert "not_an_aggregator" in exc_info.value.code, (
                "Exception should indicate that the signer is not an aggregator"
            )

            # Verify that the swarm.is_aggregator was called with correct parameters
            swarm_mock.is_aggregator.assert_called_once()
            call_args = swarm_mock.is_aggregator.call_args
            assert call_args[0][0] == account.address, (
                "Should be called with account address"
            )

    def test_model_notary_mod_no_account_config_parse_exception(
        self,
        filled_context: Context,
        call_next,
        train_message: Message,
        minimal_domain: EIP712DomainMinimal,
        caplog,
    ) -> None:
        """Test that ParseException is raised when parsing result.content with no account config."""
        # Create context without account config
        del filled_context.state.config_records[ACCOUNT_CONFIG_STATE_KEY]

        # Mock swarm methods
        swarm_mock = Mock(spec=Swarm)
        mock_config = Mock(spec=SwarmConfig)
        mock_config.get.return_value = swarm_mock
        swarm_mock.is_aggregator.return_value = True
        swarm_mock.get_eip712_domain.return_value = minimal_domain

        with patch("rizemind.swarm.config.SwarmConfig", return_value=mock_config):
            result = model_notary_mod(train_message, filled_context, call_next)

            assert isinstance(result, Message), "Should return a Message"

            with pytest.raises(ParseException):
                fit_res = recorddict_to_fitres(result.content, False)
                parse_model_notary_config(fit_res.metrics)
