from typing import Protocol

from eth_typing import ChecksumAddress
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from web3 import Web3

from rizemind.authentication.signature import recover_model_signer
from rizemind.contracts.erc.erc5267.typings import EIP712Domain
from rizemind.exception.base_exception import RizemindException


class CannotTrainException(RizemindException):
    def __init__(self, address: str) -> None:
        message = f"{address} cannot train"
        super().__init__(code="cannot_train", message=message)


class SupportsEthAccountStrategy(Protocol):
    def can_train(self, trainer: ChecksumAddress, round_id: int) -> bool: ...
    def get_eip712_domain(self) -> EIP712Domain: ...


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
    model: SupportsEthAccountStrategy
    address: str

    def __init__(
        self,
        strat: Strategy,
        model: SupportsEthAccountStrategy,
    ):
        super().__init__()
        self.strat = strat
        self.model = model
        domain = self.model.get_eip712_domain()
        self.address = domain.verifyingContract

    def initialize_parameters(self, client_manager):
        return self.strat.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        client_instructions = self.strat.configure_fit(
            server_round, parameters, client_manager
        )
        # contract_address is used in signing client
        for _, fit_ins in client_instructions:
            fit_ins.config["contract_address"] = self.address
            fit_ins.config["current_round"] = server_round
        return client_instructions

    def aggregate_fit(self, server_round, results, failures):
        whitelisted: list[tuple[ClientProxy, FitRes]] = []
        for client, res in results:
            signer = self._recover_signer(res, server_round)
            if self.model.can_train(signer, server_round):
                res.metrics["trainer_address"] = signer
                whitelisted.append((client, res))
            else:
                failures.append(CannotTrainException(signer))
        return self.strat.aggregate_fit(server_round, whitelisted, failures)

    def _recover_signer(self, res: FitRes, server_round: int):
        vrs = (
            ensure_bytes(res.metrics.get("v")),
            ensure_bytes(res.metrics.get("r")),
            ensure_bytes(res.metrics.get("s")),
        )
        eip712_domain = self.model.get_eip712_domain()
        signer = recover_model_signer(
            model=res.parameters,
            version=eip712_domain.version,
            chainid=eip712_domain.chainId,
            contract=eip712_domain.verifyingContract,
            name=eip712_domain.name,
            round=server_round,
            signature=vrs,
        )
        return Web3.to_checksum_address(signer)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.strat.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.strat.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        return self.strat.evaluate(server_round, parameters)


def ensure_bytes(value) -> bytes:
    if value is None:
        raise ValueError("Value must not be None")
    if isinstance(value, bytes):
        return value
    if isinstance(value, bool | int | float | str):
        return str(value).encode("utf-8")
    raise ValueError(f"Cannot convert value of type {type(value)} to bytes")
