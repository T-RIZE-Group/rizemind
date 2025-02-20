from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import FitRes
from rizemind.authentication.signature import recover_model_signer
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1


class CannotTrainException(BaseException):
    def __init__(self, address: str) -> None:
        message = f"{address} cannot train"
        super().__init__(message)


class EthAccountStrategy(Strategy):
    strat: Strategy
    model: ModelRegistryV1
    address: str

    def __init__(
        self,
        strat: Strategy,
        model: ModelRegistryV1,
    ):
        super().__init__()
        self.strat = strat
        self.model = model
        self.address = self.model.fl_contract.address

    def initialize_parameters(self, client_manager):
        return self.strat.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        client_instructions = self.strat.configure_fit(
            server_round, parameters, client_manager
        )
        # We need to add contract address and server round to FitIns so that clients have
        # access to it
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
        return signer

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
    if isinstance(value, (bool, int, float, str)):
        return str(value).encode("utf-8")
    raise ValueError(f"Cannot convert value of type {type(value)} to bytes")
