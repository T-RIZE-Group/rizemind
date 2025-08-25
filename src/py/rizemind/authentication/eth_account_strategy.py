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
    def __init__(self, address: str) -> None:
        message = f"{address} cannot train"
        super().__init__(code="cannot_train", message=message)


class CannotRecoverSignerException(RizemindException):
    def __init__(
        self,
    ) -> None:
        super().__init__(code="cannot_recover_signer", message="Cannot recover signer")


class EthAccountStrategy(Strategy):
    """
    A federated learning strategy that verifies model authenticity using Ethereum-based signatures.

    This class wraps an existing Flower Strategy and ensures that only authorized clients
    can contribute training updates. It does so by verifying cryptographic signatures against
    a blockchain-based model registry. If a client is not authorized, it is added to the
    failures list with a :class:`CannotTrainException`.

    :param strat: The base Flower Strategy to wrap.
    :type strat: Strategy
    :param model: The blockchain-based model registry.
    :type model: ModelRegistryV1

    **Example Usage:**

    .. code-block:: python

        strategy = SomeBaseStrategy()
        model_registry = SwarmV1.from_address(address="0xMY_MODEL_ADDRESS")
        eth_strategy = EthAccountStrategy(strategy, model_registry)
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
        super().__init__()
        self.strat = strat
        self.swarm = swarm
        domain = self.swarm.get_eip712_domain()
        self.address = domain.verifyingContract
        self.account = account

    def initialize_parameters(self, client_manager):
        return self.strat.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
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
        notary_config = parse_model_notary_config(res.metrics)
        eip712_domain = self.swarm.get_eip712_domain()
        return recover_model_signer(
            model=res.parameters,
            domain=eip712_domain,
            round=server_round,
            signature=notary_config.signature,
        )

    def configure_evaluate(self, server_round, parameters, client_manager):
        auth_cm = AuthenticatedClientManager(client_manager, server_round, self.swarm)
        return self.strat.configure_evaluate(server_round, parameters, auth_cm)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.strat.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        return self.strat.evaluate(server_round, parameters)
