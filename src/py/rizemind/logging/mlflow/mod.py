import time
from logging import WARNING
from typing import cast

import mlflow
import pandas as pd
from flwr.client.typing import ClientAppCallable
from flwr.common import Context, log
from flwr.common.constant import MessageType
from flwr.common.message import Message
from flwr.common.recorddict_compat import recorddict_to_fitres
from mlflow.entities import RunStatus, ViewType

from rizemind.logging.mlflow.config import MLFlowConfig
from rizemind.logging.train_metric_history import (
    TRAIN_METRIC_HISTORY_KEY,
    TrainMetricHistory,
)


def mlflow_mod(msg: Message, ctx: Context, call_next: ClientAppCallable) -> Message:
    """Logs metrics on an incoming TRAIN message to an Mlflow server.

    The `mlflow_mod` relies on the `TRAIN_METRIC_HISTORY_KEY` as a standardized
    metric type and reads the content of this metric for logging.
    In addition to the metrics available in `TRAIN_METRIC_HISTORY_KEY`,
    `mlflow_mod` automatically logs training_time and epochs.

    Args:
        msg: The incoming message from the ServerApp to the ClientApp.
        ctx: Context of the run.
        call_next: The next callable in the chain to process the message.

    Returns:
        The response message sent from the ClientApp to the ServerApp.
    """
    start_time = time.time()
    reply: Message = call_next(msg, ctx)
    time_diff = time.time() - start_time

    mlflow_config = MLFlowConfig.from_context(ctx=ctx)
    if mlflow_config is None:
        log(
            level=WARNING,
            msg="mlflow config was not found in client context, skipping logging.",
        )
        return reply

    mlflow.set_tracking_uri(mlflow_config.mlflow_uri)
    mlflow_experiment_name = mlflow_config.experiment_name
    mlflow_run_name = f"{mlflow_config.run_name}_client_id_{ctx.node_id}"

    if msg.metadata.message_type == MessageType.TRAIN:
        mlflow.set_experiment(experiment_name=mlflow_experiment_name)

        runs_df = cast(
            pd.DataFrame,
            mlflow.search_runs(
                experiment_names=[mlflow_experiment_name],
                filter_string=f"tags.mlflow.runName = '{mlflow_run_name}'",
                run_view_type=ViewType.ALL,
                order_by=["attributes.end_time DESC"],
                max_results=1,
            ),
        )
        epochs_passed = 0
        run_id = ""
        if runs_df.empty:
            # If a previous run doesn't exist
            # start a run with the given name
            mlflow.start_run(run_name=mlflow_run_name)
        else:
            # If a previous run exists
            # update the number of epochs passed
            epochs_passed = int(cast(int, runs_df.loc[0, "metrics.epochs"]))

            # continue the run
            run_id: str = cast(str, runs_df.loc[0, "run_id"])
            mlflow.start_run(run_id=run_id)

        if not reply.has_content():
            mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))
        else:
            # Log training time
            server_round = int(msg.metadata.group_id)
            mlflow.log_metric(key="training_time", value=time_diff, step=server_round)

            # Get metrics and log them
            fit_res = recorddict_to_fitres(reply.content, keep_input=True)
            serialized_train_metric_history = cast(
                str, fit_res.metrics.get(TRAIN_METRIC_HISTORY_KEY)
            )
            train_metric_history = TrainMetricHistory.deserialize(
                serialized_train_metric_history=serialized_train_metric_history
            )
            epochs_this_round = 0
            for metric, phases in train_metric_history.model_dump().items():
                for phase, values in phases.items():
                    for step, metric_value in enumerate(values):
                        mlflow.log_metric(
                            key=f"{phase}_{metric}",
                            value=metric_value,
                            step=step + epochs_passed,
                        )
                    epochs_this_round = max(epochs_this_round, len(values))

            epochs_passed += epochs_this_round
            mlflow.log_metric(key="epochs", value=epochs_passed)
            mlflow.end_run(status=RunStatus.to_string(RunStatus.FINISHED))

    return reply
