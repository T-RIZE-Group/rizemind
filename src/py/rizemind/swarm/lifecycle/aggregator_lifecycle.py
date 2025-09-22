import time
import timeit
from collections.abc import Sequence
from logging import INFO

from flwr.common import Context, log
from flwr.server import Grid, LegacyContext
from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, Key
from rizemind.swarm.lifecycle.aggregator_phase import AggregatorPhase
from rizemind.swarm.swarm import Swarm


def find_start_phase(phases: Sequence[AggregatorPhase]) -> AggregatorPhase:
    for phase in phases:
        if phase.can_execute():
            return phase
    raise ValueError("No start phase found")


class AggregatorLifecycle:
    """
    The lifecycle is phases pointing to each others forming a graph.
    """

    current_phase: AggregatorPhase
    swarm: Swarm
    _phases: Sequence[AggregatorPhase]
    _continue_run: bool

    def __init__(self, swarm: Swarm, phases: Sequence[AggregatorPhase]):
        self.swarm = swarm
        self.current_phase = find_start_phase(phases)
        self._phases = phases
        self._continue_run = True

    def stop(self, signum, frame):
        """TODO: propagate the signal to the phases"""
        self._continue_run = False

    def run(self, grid: Grid, context: Context):
        if not isinstance(context, LegacyContext):
            raise TypeError(
                f"Expect a LegacyContext, but get {type(context).__name__}."
            )
        while self.continue_run():
            current_phase = self.current_phase
            if self.current_phase.can_execute():
                start_time = timeit.default_timer()
                current_round = self.swarm.get_current_round()
                log(INFO, "")
                log(
                    INFO,
                    "[ROUND %s] [Lifecycle: executing %s]",
                    current_round,
                    type(current_phase).__name__,
                )
                context.state.config_records[MAIN_CONFIGS_RECORD][Key.CURRENT_ROUND] = (
                    current_round
                )
                self.current_phase = self.current_phase.execute(grid, context)
                end_time = timeit.default_timer()
                elapsed = end_time - start_time
                log(
                    INFO,
                    "[ROUND %s] [Lifecycle: executed %s in %.2fs]",
                    current_round,
                    type(current_phase).__name__,
                    elapsed,
                )
                if self.current_phase is not None:
                    log(
                        INFO,
                        "[ROUND %s] [Lifecycle: entering %s]",
                        current_round,
                        type(self.current_phase).__name__,
                    )
                else:
                    log(
                        INFO,
                        "[Lifecycle: completed]",
                    )
                    self.current_phase = find_start_phase(self._phases)
            else:
                time.sleep(0.5)

    def continue_run(self) -> bool:
        return self._continue_run
