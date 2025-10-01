from logging import INFO
from typing import cast

from eth_typing import Address
from flwr.common import EvaluateIns, EvaluateRes, FitIns, Parameters
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from web3 import Web3

from rizemind.strategies.compensation.typings import SupportsDistribute


class SimpleCompensationStrategy(Strategy):
    """A federated learning strategy with equal compensation distribution.

    This strategy acts as a decorator around an existing Flower strategy, adding
    compensation functionality that distributes equal rewards (1.0) to all
    participating clients after each training round. The compensation is distributed
    through blockchain.

    Attributes:
        strategy: The underlying federated learning strategy to delegate operations to.
        model: The reward distributor.
    """

    strategy: Strategy
    swarm: SupportsDistribute

    def __init__(self, strategy: Strategy, model: SupportsDistribute) -> None:
        """Initialize the simple compensation strategy.

        Args:
            strategy: The base federated learning strategy to wrap.
            model: The reward distributor.
        """
        self.strategy = strategy
        self.swarm = model

    def calculate(self, client_ids: list[Address]):
        """Compensate each client equally.

        This method implements a simple equal compensation scheme where all
        participating clients receive the same reward score of 1.0.

        Args:
            client_ids: List of client blockchain addresses that participated in training.

        Returns:
            List of tuples containing checksum addresses and their corresponding
            compensation scores (all equal to 1.0).
        """
        log(INFO, "calculate: calculating compensations.")
        return [(Web3.to_checksum_address(id), 1.0) for id in client_ids]

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results and distribute compensation to participants.

        This method processes training results from clients, calculates compensation
        scores using the simple equal distribution scheme, distributes the rewards,
        and then delegates the actual model aggregation to the underlying strategy.

        Args:
            server_round: Current federated learning round number.
            results: List of training results from participating clients.
            failures: List of failed training attempts.

        Returns:
            Aggregated model parameters and metrics from the underlying strategy.
        """
        log(
            INFO,
            "aggregate_fit: training results received from the clients",
        )
        log(INFO, "aggregate_fit: initializing aggregation")
        trainer_scores = self.calculate(
            [cast(Address, res.metrics["trainer_address"]) for _, res in results]
        )
        self.swarm.distribute(server_round, trainer_scores)
        return self.strategy.aggregate_fit(server_round, results, failures)

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialize global model parameters for federated learning.

        Delegates the parameter initialization to the underlying strategy while
        logging the start of the training phase.

        Args:
            client_manager: Manager handling available clients.

        Returns:
            Initial model parameters, or None if not applicable.
        """
        log(
            INFO,
            "initialize_parameters: first training phase started",
        )
        log(
            INFO,
            "initialize_parameters: initializing model parameters for the first time",
        )
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure client training instructions for the current round.

        Delegates the configuration of training instructions to the underlying strategy.

        Args:
            server_round: Current federated learning round number.
            parameters: Current global model parameters.
            client_manager: Manager handling available clients.

        Returns:
            List of tuples containing client proxies and their training instructions.
        """
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure client evaluation instructions for the current round.

        Delegates the configuration of evaluation instructions to the underlying strategy.

        Args:
            server_round: Current federated learning round number.
            parameters: Current global model parameters.
            client_manager: Manager handling available clients.

        Returns:
            List of tuples containing client proxies and their evaluation instructions.
        """
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate evaluation results from clients.

        Delegates the aggregation of evaluation results to the underlying strategy.

        Args:
            server_round: Current federated learning round number.
            results: List of evaluation results from participating clients.
            failures: List of failed evaluation attempts.

        Returns:
            Tuple containing aggregated loss and metrics dictionary.
        """
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate the global model on the server side.

        Delegates the server-side evaluation to the underlying strategy.

        Args:
            server_round: Current federated learning round number.
            parameters: Current global model parameters to evaluate.

        Returns:
            Tuple containing loss and metrics, or None if evaluation is not performed.
        """
        return self.strategy.evaluate(server_round, parameters)
