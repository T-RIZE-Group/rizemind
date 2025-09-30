from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class AlwaysTrueCriterion(Criterion):
    """A criterion that marks all clients as eligible for sampling."""

    def select(self, client: ClientProxy) -> bool:
        """Returns True for any client, marking it as eligible for sampling.

        Args:
            client: The client whose eligibility is being determined.
        """
        return True


class AndCriterion(Criterion):
    """A criterion that performs logical AND operation on two criteria.

    This criterion evaluates both provided criteria in parallel using a thread pool
    and returns True only if both criteria evaluate to True for a given client.

    Attributes:
        criterion_a: First criterion to evaluate. If None, defaults to AlwaysTrueCriterion.
        criterion_b: Second criterion to evaluate. If None, defaults to AlwaysTrueCriterion.
    """

    criterion_a: Criterion
    criterion_b: Criterion

    _pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)

    def __init__(self, criterion_a: Criterion | None, criterion_b: Criterion | None):
        """Initialize the AndCriterion with two criteria.

        Args:
            criterion_a: First criterion to evaluate. If None, defaults to AlwaysTrueCriterion.
            criterion_b: Second criterion to evaluate. If None, defaults to AlwaysTrueCriterion.
        """
        self.criterion_a = (
            criterion_a if criterion_a is not None else AlwaysTrueCriterion()
        )
        self.criterion_b = (
            criterion_b if criterion_b is not None else AlwaysTrueCriterion()
        )

    def select(self, client: ClientProxy) -> bool:
        """Evaluate both criteria and return True only if both pass.

        The criteria are evaluated in parallel using a thread pool for efficiency.
        Any exceptions from either criterion will be propagated.

        Args:
            client: The client to evaluate against both criteria.

        Returns:
            True if both criteria evaluate to True, False otherwise.
        """
        future_a = self._pool.submit(self.criterion_a.select, client)
        future_b = self._pool.submit(self.criterion_b.select, client)

        # Propagate exceptions (if any) and combine results
        return future_a.result() and future_b.result()


class ClientManagerWithCriterion(ClientManager):
    """Wraps another ClientManager and injects authentication Criterion.

    Attributes:
        round_id: Current federated learning round identifier.
        swarm: Current swarm.
    """

    round_id: int
    criterion: Criterion

    def __init__(
        self, base_manager: ClientManager, round_id: int, criterion: Criterion
    ) -> None:
        """Initialize the authenticated client manager.

        Args:
            base_manager: The underlying client manager to wrap with authentication.
            round_id: Current federated learning round identifier.
            swarm: Swarm protocol instance that supports Ethereum account strategy for authentication.
        """
        self._base = base_manager
        self.round_id = round_id
        self.criterion = criterion

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        criterion: Any | None = None,
    ) -> list[ClientProxy]:
        """Sample clients with authentication checks.

        Adds authentication criterion to ensure only clients that can train
        in the current round are selected. The authentication criterion is
        combined with any provided criterion using logical AND.

        Args:
            num_clients: Number of clients to sample.
            min_num_clients: Minimum number of clients required. Defaults to None.
            criterion: Additional criterion to apply. Defaults to None.

        Returns:
            List of authenticated client proxies that meet all criteria.
        """
        clients = self._base.sample(
            num_clients,
            min_num_clients,
            AndCriterion(self.criterion, criterion),
        )
        return clients

    def num_available(self) -> int:
        """Get the number of available clients.

        Returns:
            The total number of clients available in the base manager.
        """
        return self._base.num_available()

    def register(self, client: ClientProxy) -> bool:
        """Register a client with the base manager.

        Args:
            client: The client proxy to register.

        Returns:
            True if registration was successful, False otherwise.
        """
        return self._base.register(client)

    def unregister(self, client: ClientProxy) -> None:
        """Unregister a client from the base manager.

        Args:
            client: The client proxy to unregister.
        """
        return self._base.unregister(client)

    def all(self) -> dict[str, ClientProxy]:
        """Get all registered clients.

        Returns:
            Dictionary mapping client IDs to their corresponding client proxies.
        """
        return self._base.all()

    def wait_for(
        self,
        num_clients: int,
        timeout: int,
    ) -> bool:
        """Wait for a minimum number of clients to be available.

        Args:
            num_clients: Minimum number of clients to wait for.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if the required number of clients became available within the timeout, False otherwise.
        """
        return self._base.wait_for(num_clients, timeout)

    def __getattr__(self, name: str):
        return getattr(self._base, name)
