from logging import ERROR

from flwr.common import Context, log
from flwr.server import Grid
from flwr.server.typing import Workflow
from rizemind.contracts.base_contract import RizemindContractException
from rizemind.contracts.swarm.training.base_training_phase.phases import (
    EVALUATION_PHASE,
    EVALUATOR_REGISTRATION_PHASE,
    IDLE_PHASE,
)
from rizemind.swarm.lifecycle.aggregator_lifecycle import AggregatorLifecycle
from rizemind.swarm.lifecycle.aggregator_phase import AggregatorPhase
from rizemind.swarm.lifecycle.workflow_phase import WorkflowPhase
from rizemind.swarm.swarm import Swarm


class IdlePhase(AggregatorPhase):
    """
    Idle phase.
    """

    swarm: Swarm
    _next_phase: AggregatorPhase

    def __init__(self, swarm: Swarm, next_phase: AggregatorPhase):
        super().__init__()
        self.swarm = swarm
        self._next_phase = next_phase

    def can_execute(self) -> bool:
        current_phase = self.swarm.get_current_phase()
        return current_phase == IDLE_PHASE or current_phase == EVALUATION_PHASE

    def execute(self, grid: Grid, context: Context) -> AggregatorPhase:
        while not self.swarm.can_start_training_round():
            pass
        self.swarm.start_training_round()
        return self.next_phase(grid, context)

    def next_phase(self, grid: Grid, context: Context) -> AggregatorPhase:
        return self._next_phase


class TrainingPhase(WorkflowPhase):
    """
    Training phase.
    """

    swarm: Swarm
    _next_phase: AggregatorPhase

    def __init__(
        self, swarm: Swarm, fit_workflow: Workflow, next_phase: AggregatorPhase
    ):
        super().__init__(fit_workflow)
        self.swarm = swarm
        self._next_phase = next_phase

    def can_execute(self) -> bool:
        return self.swarm.is_training()

    def execute(self, grid: Grid, context: Context) -> WorkflowPhase:
        try:
            return super().execute(grid, context)
        except RizemindContractException as e:
            log(ERROR, e)
        return self

    def next_phase(self, grid: Grid, context: Context) -> AggregatorPhase:
        return self._next_phase


class EvaluatorRegistrationPhase(AggregatorPhase):
    """
    Evaluator registration phase.
    """

    swarm: Swarm
    _next_phase: AggregatorPhase

    def __init__(self, swarm: Swarm, next_phase: AggregatorPhase):
        super().__init__()
        self.swarm = swarm
        self._next_phase = next_phase

    def can_execute(self) -> bool:
        return self.swarm.get_current_phase() == EVALUATOR_REGISTRATION_PHASE

    def execute(self, grid: Grid, context: Context) -> AggregatorPhase:
        while True:
            phase = self.swarm.get_current_phase()
            if phase != EVALUATOR_REGISTRATION_PHASE:
                self.swarm.update_phase()
                return self.next_phase(grid, context)

    def next_phase(self, grid: Grid, context: Context) -> AggregatorPhase:
        return self._next_phase


class EvaluationPhase(WorkflowPhase):
    """
    Training phase.
    """

    swarm: Swarm
    _next_phase: AggregatorPhase | None

    def __init__(
        self,
        swarm: Swarm,
        evaluation_workflow: Workflow,
        next_phase: AggregatorPhase | None,
    ):
        super().__init__(evaluation_workflow)
        self.swarm = swarm
        self._next_phase = next_phase

    def can_execute(self) -> bool:
        return self.swarm.get_current_phase() == EVALUATION_PHASE

    def next_phase(self, grid: Grid, context: Context) -> AggregatorPhase | None:
        return self._next_phase


class BaseTrainingLifecycle(AggregatorLifecycle):
    """
    Base training phase.
    """

    def __init__(
        self,
        *,
        fit_workflow: Workflow,
        centralized_evaluate_workflow: Workflow,
        evaluate_workflow: Workflow,
        swarm: Swarm,
    ):
        evaluation_phase = EvaluationPhase(swarm, evaluate_workflow, None)
        evaluator_registration_phase = EvaluatorRegistrationPhase(
            swarm,
            evaluation_phase,
        )
        training_phase = TrainingPhase(
            swarm,
            fit_workflow,
            evaluator_registration_phase,
        )
        idle_phase = IdlePhase(
            swarm,
            training_phase,
        )
        super().__init__(
            swarm,
            [
                idle_phase,
                training_phase,
                evaluator_registration_phase,
                evaluation_phase,
            ],
        )
