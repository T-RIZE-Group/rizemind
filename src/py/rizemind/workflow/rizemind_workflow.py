import io
import timeit
from logging import INFO

from flwr.common import (
    ConfigRecord,
    Context,
    log,
)
from flwr.server.compat.app_utils import start_update_client_manager_thread
from flwr.server.compat.legacy_context import LegacyContext
from flwr.server.grid import Grid
from flwr.server.typing import Workflow
from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, Key
from flwr.server.workflow.default_workflows import (
    default_centralized_evaluation_workflow,
    default_evaluate_workflow,
    default_fit_workflow,
    default_init_params_workflow,
)
from rizemind.swarm.lifecycle.aggregator_lifecycle import AggregatorLifecycle
from rizemind.swarm.lifecycle.base_training_phase import BaseTrainingLifecycle
from rizemind.swarm.swarm import Swarm


class RizemindWorkflow:
    """Default workflow in Flower."""

    fit_workflow: Workflow
    evaluate_workflow: Workflow
    swarm: Swarm
    lifecycle: AggregatorLifecycle

    def __init__(
        self,
        swarm: Swarm,
        fit_workflow: Workflow | None = None,
        evaluate_workflow: Workflow | None = None,
    ) -> None:
        self.swarm = swarm
        if fit_workflow is None:
            fit_workflow = default_fit_workflow
        if evaluate_workflow is None:
            evaluate_workflow = default_evaluate_workflow
        self.fit_workflow = fit_workflow
        self.evaluate_workflow = evaluate_workflow
        self.lifecycle = BaseTrainingLifecycle(
            fit_workflow=fit_workflow,
            centralized_evaluate_workflow=default_centralized_evaluation_workflow,
            evaluate_workflow=evaluate_workflow,
            swarm=swarm,
        )

    def __call__(self, grid: Grid, context: Context) -> None:
        """Execute the workflow."""
        if not isinstance(context, LegacyContext):
            raise TypeError(
                f"Expect a LegacyContext, but get {type(context).__name__}."
            )

        # Start the thread updating nodes
        thread, f_stop, c_done = start_update_client_manager_thread(
            grid, context.client_manager
        )

        # Wait until the node registration done
        c_done.wait()

        # Initialize parameters
        log(INFO, "[INIT]")
        default_init_params_workflow(grid, context)

        # Run federated learning for num_rounds
        start_time = timeit.default_timer()
        cfg = ConfigRecord()
        cfg[Key.START_TIME] = start_time
        context.state.config_records[MAIN_CONFIGS_RECORD] = cfg

        self.lifecycle.run(grid, context)

        # Bookkeeping and log results
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        hist = context.history
        log(INFO, "")
        log(INFO, "[SUMMARY]")
        log(
            INFO,
            "Run finished %s round(s) in %.2fs",
            context.config.num_rounds,
            elapsed,
        )
        for idx, line in enumerate(io.StringIO(str(hist))):
            if idx == 0:
                log(INFO, "%s", line.strip("\n"))
            else:
                log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        # Terminate the thread
        f_stop.set()
        thread.join()

    def should_exit(self) -> bool:
        return False
