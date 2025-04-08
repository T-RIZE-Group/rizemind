from typing import Dict
from flwr.client import Client
from flwr.common import FitRes, FitIns, EvaluateIns
from rizemind.authentication.signature import (
    sign_parameters_model,
)
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1
from eth_account.signers.base import BaseAccount
from web3 import Web3


class SigningClient:
    """
    A proxy class that wraps a Flower Client to add signing functionality for trained parameters.

    This class ensures authenticity by signing the trained model parameters before sending them
    to the aggregator. It uses Ethereum-based signing via an `eth_account` account and signs
    parameters using EIP-712 structured data.

    :param client: The Flower `Client` instance to wrap.
    :type client: Client
    :param account: The Ethereum account used for signing.
    :type account: BaseAccount
    :param w3: The Web3 instance used for interacting with blockchain contracts.
    :type w3: Web3

    **Example Usage:**

    .. code-block:: python

        from flwr.client import NumPyClient
        from web3 import Web3
        from eth_account import Account
        from rizemind.authentication.eth_account_signature import SigningClient

        client = NumPyClient()
        account = Account.create()  # or load from mnemonic/private key
        w3 = Web3()

        signed_client = SigningClient(client, account, w3)
    """

    client: Client
    account: BaseAccount
    w3: Web3

    def __init__(self, client: Client, account: BaseAccount, w3: Web3):
        """
        Initialize the SigningClient.

        :param client: The Flower Client to wrap.
        :type client: Client
        :param account: Ethereum account used for signing model parameters.
        :type account: BaseAccount
        :param w3: Web3 instance used for interacting with Ethereum contracts.
        :type w3: Web3
        """
        self.client = client
        self.account = account
        self.w3 = w3

    def __getattr__(self, name):
        return getattr(self.client, name)

    def fit(self, ins: FitIns):
        """
        Train the model using the wrapped client and sign the trained parameters.

        :param ins: Instructions for fitting the model.
        :type ins: FitIns
        :return: The signed fit results.
        :rtype: FitRes
        """
        results: FitRes = self.client.fit(ins)
        contract_address = str(ins.config["contract_address"])
        round = ensure_int(ins.config["current_round"])
        signature = self._sign(
            res=results, round=round, contract_address=contract_address
        )

        results.metrics = results.metrics | signature
        return results

    def _sign(self, res: FitRes, round: int, contract_address: str) -> Dict[str, bytes]:
        model = ModelRegistryV1.from_address(contract_address, account=None, w3=self.w3)  # type: ignore
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
