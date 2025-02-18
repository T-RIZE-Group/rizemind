from typing import Dict
from flwr.client import Client
from flwr.common import FitRes, FitIns, EvaluateIns
from rize_dml.authentication.signature import (
    sign_parameters_model,
)
from rize_dml.contracts.models.model_registry_v1 import ModelRegistryV1
from eth_account import Account


class SigningClient:
    client: Client
    account: Account

    def __init__(self, client: Client, account: Account):
        self.client = client
        self.account = account

    def __getattr__(self, name):
        return getattr(self.client, name)

    def fit(self, ins: FitIns):
        # Call the original fit method on the proxied Client
        results: FitRes = self.client.fit(ins)
        contract_address = str(ins.config["address"])
        round = ins.config["current_round"]
        round_in_int = ensure_int(round)
        signature = self._sign(
            res=results, round=round_in_int, contract_address=contract_address
        )

        results.metrics = results.metrics | signature
        return results

    def _sign(self, res: FitRes, round: int, contract_address: str) -> Dict[str, bytes]:
        model = ModelRegistryV1.from_address(contract_address, None) # type: ignore
        eip712_domain = model.get_eip712_domain()

        # Output Signer
        signature = sign_parameters_model(
            account=self.account,
            parameters=res.parameters,
            name=eip712_domain.name,
            chainid=eip712_domain.chainId,
            contract=eip712_domain.verifyingContract,
            version=eip712_domain.version,
            round=round,
        )
        return {
            "r": signature.r.to_bytes(32, byteorder="big"),
            "s": signature.s.to_bytes(32, byteorder="big"),
            "v": signature.v.to_bytes(1, byteorder="big"),
        }

    def evaluate(self, ins: EvaluateIns):
        return self.client.evaluate(ins)


def ensure_int(value) -> int:
    if value is None:
        raise ValueError("Value must not be None")
    if isinstance(value, int):
        return value
    if isinstance(value, (bool, bytes, float, str)):
        return int(value)
    raise ValueError(f"Cannot convert value of type {type(value)} to int")
