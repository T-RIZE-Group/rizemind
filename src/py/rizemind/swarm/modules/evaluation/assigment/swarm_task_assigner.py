from logging import WARNING
from eth_typing import ChecksumAddress
from flwr.common import log
from rizemind.swarm.modules.evaluation.assigment.task_assigner import (
    SupportsTaskAssignement,
)
from rizemind.swarm.swarm import Swarm


class SwarmTaskAssigner(SupportsTaskAssignement):
    _swarm: Swarm

    def __init__(self, swarm: Swarm):
        self._swarm = swarm

    def tasks_of_evaluator(
        self, evaluator: ChecksumAddress, round_id: int
    ) -> list[int]:
        try:
            node_id = self._swarm.evaluator_registry.get_evaluator_id(
                round_id, evaluator
            )
            return self._swarm.task_assignement.tasks_of_node(round_id, node_id - 1)
        except Exception as e:
            log(WARNING, f"tasks_of_evaluator: {e}")
            return []
