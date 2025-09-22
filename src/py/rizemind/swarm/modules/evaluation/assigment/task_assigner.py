from typing import Protocol

from eth_typing import ChecksumAddress
from flwr.common import EvaluateIns
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from rizemind.authentication.authenticated_client_properties import (
    AuthenticatedClientProperties,
)
from rizemind.authentication.eth_account_strategy import hash_parameters
from rizemind.strategies.contribution.shapley.trainer_set import (
    TrainerSetAggregateStore,
)
from rizemind.swarm.modules.evaluation.ins import prepare_evaluation_task_ins


class SupportsTaskAssignement(Protocol):
    def tasks_of_evaluator(
        self, evaluator: ChecksumAddress, round_id: int
    ) -> list[int]: ...


class TaskAssigner:
    _task_assignment: SupportsTaskAssignement
    _trainer_set_store: TrainerSetAggregateStore

    def __init__(
        self,
        task_assignment: SupportsTaskAssignement,
        trainer_set_store: TrainerSetAggregateStore,
    ):
        self._task_assignment = task_assignment
        self._trainer_set_store = trainer_set_store

    def tasks_of_evaluator(
        self, evaluator: ChecksumAddress, round_id: int
    ) -> list[int]:
        return self._task_assignment.tasks_of_evaluator(evaluator, round_id)

    def configure_evaluate(
        self, server_round: int, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        configurations: list[tuple[ClientProxy, EvaluateIns]] = []
        for evaluator in client_manager.all().values():
            auth = AuthenticatedClientProperties.from_client(evaluator)
            tasks = self.tasks_of_evaluator(auth.trainer_address, server_round)
            for task in tasks:
                aggregates = self._trainer_set_store.get_set_by_order(task)
                for aggregate in aggregates:
                    task_ins = prepare_evaluation_task_ins(
                        round_id=server_round,
                        eval_id=aggregate.order,
                        set_id=int(aggregate.id),
                        model_hash=hash_parameters(aggregate.parameters),
                    )
                    evaluate_ins = EvaluateIns(
                        parameters=aggregate.parameters,
                        config=task_ins,
                    )
                    configurations.append((evaluator, evaluate_ins))
        return configurations
