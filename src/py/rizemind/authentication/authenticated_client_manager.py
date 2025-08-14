from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from rizemind.authentication.can_train_criterion import CanTrainCriterion
from rizemind.authentication.typing import SupportsEthAccountStrategy


class AlwaysTrueCriterion(Criterion):
    def select(self, client: ClientProxy) -> bool:
        return True


class AndCriterion(Criterion):
    criterion_a: Criterion
    criterion_b: Criterion

    _pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)

    def __init__(self, criterion_a: Criterion | None, criterion_b: Criterion | None):
        self.criterion_a = (
            criterion_a if criterion_a is not None else AlwaysTrueCriterion()
        )
        self.criterion_b = (
            criterion_b if criterion_b is not None else AlwaysTrueCriterion()
        )

    def select(self, client: ClientProxy) -> bool:
        future_a = self._pool.submit(self.criterion_a.select, client)
        future_b = self._pool.submit(self.criterion_b.select, client)

        # Propagate exceptions (if any) and combine results
        return future_a.result() and future_b.result()


class AuthenticatedClientManager(ClientManager):
    """Wraps another ClientManager and injects authentication Criterion."""

    round_id: int
    swarm: SupportsEthAccountStrategy

    def __init__(
        self,
        base_manager: ClientManager,
        round_id: int,
        swarm: SupportsEthAccountStrategy,
    ) -> None:
        self._base = base_manager
        self.round_id = round_id
        self.swarm = swarm

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        criterion: Any | None = None,
    ) -> list[ClientProxy]:
        authenticated_criterion = CanTrainCriterion(self.round_id, self.swarm)
        clients = self._base.sample(
            num_clients,
            min_num_clients,
            AndCriterion(authenticated_criterion, criterion),
        )
        return clients

    def num_available(self) -> int:
        return self._base.num_available()

    def register(self, client: ClientProxy) -> bool:
        return self._base.register(client)

    def unregister(self, client: ClientProxy) -> None:
        return self._base.unregister(client)

    def all(self) -> dict[str, ClientProxy]:
        return self._base.all()

    def wait_for(
        self,
        num_clients: int,
        timeout: int,
    ) -> bool:
        return self._base.wait_for(num_clients, timeout)

    def __getattr__(self, name: str):
        return getattr(self._base, name)
