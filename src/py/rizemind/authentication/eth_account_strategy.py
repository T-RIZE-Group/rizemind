from eth_account.signers.base import BaseAccount
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from rizemind.authentication.authenticated_client_manager import (
    AuthenticatedClientManager,
)
from rizemind.authentication.authenticated_client_properties import (
    AuthenticatedClientProperties,
)
from rizemind.authentication.notary.model.config import (
    parse_model_notary_config,
    prepare_model_notary_config,
)
from rizemind.authentication.notary.model.model_signature import (
    hash_parameters,
    recover_model_signer,
    sign_parameters_model,
)
from rizemind.authentication.typing import SupportsEthAccountStrategy
from rizemind.exception.base_exception import RizemindException
from rizemind.exception.parse_exception import ParseException


class CannotTrainException(RizemindException):
    """An attempt was made to train with an unauthorized address."""

    def __init__(self, address: str) -> None:
        message = f"{address} cannot train"
        super().__init__(code="cannot_train", message=message)


class CannotRecoverSignerException(RizemindException):
    """The signer of a model update could not be recovered."""

    def __init__(
        self,
    ) -> None:
        super().__init__(code="cannot_recover_signer", message="Cannot recover signer")


class EthAccountStrategy(Strategy):
    """A federated learning strategy that verifies model authenticity.

    This strategy wraps an existing Flower Strategy to ensure that only authorized
    clients can contribute training updates. It verifies cryptographic signatures
    against a blockchain-based model registry. If a client is not authorized, it
    is added to the failures list with a `CannotTrainException`.

    Attributes:
        strat: The base Flower Strategy to wrap.
        swarm: The blockchain-based model registry.
        address: The contract address of the swarm.
        account: The Ethereum account used for signing.

    Example Usage:

    ```python
        strategy = SomeBaseStrategy()
        model_registry = SwarmV1.from_address(address="0xMY_MODEL_ADDRESS")
        eth_strategy = EthAccountStrategy(strategy, model_registry)
    ```
    """

    strat: Strategy
    swarm: SupportsEthAccountStrategy
    address: str
    account: BaseAccount

    def __init__(
        self,
        strat: Strategy,
        swarm: SupportsEthAccountStrategy,
        account: BaseAccount,
    ):
        """Initializes the EthAccountStrategy.

        Args:
            strat: The base Flower Strategy to wrap.
            swarm: The blockchain-based model registry.
            account: The Ethereum account used for signing.
        """
        super().__init__()
        self.strat = strat
        self.swarm = swarm
        domain = self.swarm.get_eip712_domain()
        self.address = domain.verifyingContract
        self.account = account

    def initialize_parameters(self, client_manager):
        """Initializes model parameters."""
        return self.strat.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        """Prepare fit instructions and attach notary metadata.

        Wraps the base strategy's `configure_fit` to use
        `AuthenticatedClientManager` and appends a signed notary payload to each
        client's config so that clients can sign their updates.

        Args:
            server_round: The current server round.
            parameters: The global model parameters to send to clients.
            client_manager: The Flower client manager.

        Returns:
            The list of client instructions produced by the wrapped strategy
            with notary metadata attached to each instruction's config.
        """
        auth_cm = AuthenticatedClientManager(client_manager, server_round, self.swarm)
        client_instructions = self.strat.configure_fit(
            server_round, parameters, auth_cm
        )
        domain = self.swarm.get_eip712_domain()
        for _, fit_ins in client_instructions:
            signature = sign_parameters_model(
                account=self.account,
                domain=domain,
                parameters=fit_ins.parameters,
                round=server_round,
            )
            notary_config = prepare_model_notary_config(
                round_id=server_round,
                domain=domain,
                signature=signature,
                model_hash=hash_parameters(fit_ins.parameters),
            )
            fit_ins.config = fit_ins.config | notary_config
        return client_instructions

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results from authorized clients only.

        Recovers the signer address from each client's fit result, tags the
        client with the recovered address, and filters out contributions from
        non-authorized addresses. Unauthorized attempts are recorded as
        failures. Delegates final aggregation to the wrapped strategy.

        Args:
            server_round: The current server round.
            results: A list of tuples `(ClientProxy, FitRes)` received from
                clients.
            failures: A list that will be extended with failures that occur
                during processing.

        Returns:
            The aggregated result as returned by the wrapped strategy's
            `aggregate_fit`.
        """
        whitelisted: list[tuple[ClientProxy, FitRes]] = []
        for client, res in results:
            try:
                signer = self._recover_signer(res, server_round)
                properties = AuthenticatedClientProperties(trainer_address=signer)
                properties.tag_client(client)
                if self.swarm.can_train(signer, server_round):
                    whitelisted.append((client, res))
                else:
                    failures.append(CannotTrainException(signer))
            except ParseException:
                failures.append(CannotRecoverSignerException())
        return self.strat.aggregate_fit(server_round, whitelisted, failures)

    def _recover_signer(self, res: FitRes, server_round: int):
        """Recovers the signer's address from a client's response.

        Args:
            res: The client's fit response.
            server_round: The current server round.

        Returns:
            The Ethereum address of the signer.

        Raises:
            ParseException: If the notary configuration cannot be parsed.
        """
        notary_config = parse_model_notary_config(res.metrics)
        eip712_domain = self.swarm.get_eip712_domain()
        return recover_model_signer(
            model=res.parameters,
            domain=eip712_domain,
            round=server_round,
            signature=notary_config.signature,
        )

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Prepare evaluation instructions using authenticated client manager.

        Wraps the base strategy's `configure_evaluate` with
        `AuthenticatedClientManager` and delegates to the wrapped strategy.

        Args:
            server_round: The current server round.
            parameters: The global model parameters to send to clients.
            client_manager: The Flower client manager.

        Returns:
            The list of client evaluation instructions produced by the wrapped
            strategy.
        """
        auth_cm = AuthenticatedClientManager(client_manager, server_round, self.swarm)
        return self.strat.configure_evaluate(server_round, parameters, auth_cm)

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results by delegating to the wrapped strategy.

        Args:
            server_round: The current server round.
            results: A list of tuples `(ClientProxy, EvaluateRes)` received
                from clients.
            failures: A list of failures encountered during evaluation.

        Returns:
            The aggregated evaluation result as returned by the wrapped
            strategy's `aggregate_evaluate`.
        """
        return self.strat.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        """Evaluate the current global model via the wrapped strategy.

        Args:
            server_round: The current server round.
            parameters: The global model parameters to evaluate.

        Returns:
            The evaluation result as returned by the wrapped strategy's
            `evaluate`.
        """
        return self.strat.evaluate(server_round, parameters)
