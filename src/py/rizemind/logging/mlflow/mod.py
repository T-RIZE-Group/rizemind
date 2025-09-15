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

###
# Mlflow Mod Organization
#
# Mlflow requires two parameters to distinguish each run:
# 1. Experiment Name: Which is the name given to the whole operation. Many
# runs are performed under the same experiment name, and usually the dataset
# and model architecture remains the same in an experiment.
# 2. Run Name: Which is a single name given to a full execution of a training
# cycle. After the end of each run, it is expected that the model is fully tr
# -ained.
#
# Therefore, for the name of the experiment, we chose "RealMLP-Federated" to
# indicate the model and style of the training operation. This value is shared
# among the clients, but the Run ID must be persistent between each run, there
# -fore it is an information that must be sent over by the server, since client's
# are stateless in flower.
###


def mlflow_mod(msg: Message, ctx: Context, app: ClientAppCallable) -> Message:
    start_time = time.time()
    reply: Message = app(msg, ctx)
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
