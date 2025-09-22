from typing import Self

from flwr.common import Context
from flwr.server import Grid
from flwr.server.typing import Workflow
from rizemind.swarm.lifecycle.aggregator_phase import AggregatorPhase


class WorkflowPhase(AggregatorPhase):
    _workflow: Workflow

    def __init__(self, workflow: Workflow):
        self._workflow = workflow

    def execute(self, grid: Grid, context: Context) -> Self:
        self._workflow(grid, context)
        return self.next_phase(grid, context)
