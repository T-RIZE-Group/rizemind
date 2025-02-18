from abc import abstractmethod
from typing import Any
from eth_typing import Address
from rize_dml.authentication.eth_account_strategy import EthAccountStrategy
from flwr.server.strategy import Strategy


class CompensationStrategy(Strategy):
    strategy: EthAccountStrategy

    def __init__(self, strategy: EthAccountStrategy) -> None:
        "Takes a strategy"
        self.strategy = strategy

    @abstractmethod
    def calculate(self, client_ids: list[Address]) -> tuple[list[Address], list[Any]]:
        "Calculates and returns the compensation for each client in the list"
