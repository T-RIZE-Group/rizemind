import statistics
from logging import DEBUG
from typing import cast

from dotenv import load_dotenv
from flwr.common import Context, log, ndarrays_to_parameters
from flwr.common.typing import Metrics, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_strategy import EthAccountStrategy
from rizemind.configuration.toml_config import TomlConfig
from rizemind.contracts.compensation.shapley.decentralized.shapley_value_strategy import (
    DecentralShapleyValueStrategy,
)
from rizemind.contracts.compensation.shapley.shapley_value_strategy import Coalition
from rizemind.contracts.models.model_factory_v1 import (
    ModelFactoryV1,
    ModelFactoryV1Config,
)
from rizemind.web3.config import Web3Config

from tabpfn_local_example.task import get_weights, load_model
import polars as pl


def weighted_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    num_examples = [num_examples for num_examples, _ in metrics]
    log(DEBUG, "Printing metrics available")
    log(DEBUG, metrics)
    rmses = [num_examples * float(m["rmse"]) for num_examples, m in metrics]
    maes = [num_examples * float(m["mae"]) for num_examples, m in metrics]
    r2_scores = [num_examples * float(m["r2_score"]) for num_examples, m in metrics]
    return {
        "r2_score": sum(r2_scores) / sum(num_examples),
        "rmse": sum(rmses) / sum(num_examples),
        "mae": sum(maes) / sum(num_examples),
    }


def aggregate_coalitions(coalitions: list[Coalition]) -> dict[str, Scalar]:
    rmses = [float(coalition.get_metric("rmse", 0)) for coalition in coalitions]
    maes = [float(coalition.get_metric("mae", 0)) for coalition in coalitions]
    r2_scores = [float(coalition.get_metric("r2_score", 0)) for coalition in coalitions]
    return {
        "avg_root_mean_squared_error": statistics.mean(rmses),
        "median_root_mean_squared_error": statistics.median(rmses),
        "avg_mean_absolute_error": statistics.mean(maes),
        "median_mean_absolute_error": statistics.median(maes),
        "avg_r2_score": statistics.mean(r2_scores),
        "median_r2_score": statistics.median(r2_scores),
    }


def server_fn(context: Context):
    load_dotenv()
    config = TomlConfig("./pyproject.toml")
    sample_data_path = cast(str, config.get("tool.dataset.config.sample_data_path"))
    label_name = cast(str, config.get("tool.dataset.config.label_name"))
    df_sample = pl.read_csv(sample_data_path)
    X_sample, y_sample = (
        df_sample.drop(label_name).to_pandas(),
        df_sample.select(pl.col(label_name)).to_numpy().ravel(),
    )
    model = load_model(X_sample, y_sample)
    model_weights = get_weights(model.model_)
    parameters = ndarrays_to_parameters(model_weights)

    fedavg_strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        # fit_metrics_aggregation_fn=weighted_metrics,
        evaluate_metrics_aggregation_fn=weighted_metrics,
    )

    num_rounds = int(context.run_config["num-server-rounds"])
    auth_config = AccountConfig(**config.get("tool.eth.account"))
    web3_config = Web3Config(**config.get("tool.web3"))
    w3 = web3_config.get_web3()
    account = auth_config.get_account(0)
    members = []
    for i in range(1, 11):
        trainer = auth_config.get_account(i)
        members.append(trainer.address)

    model_v1_config = ModelFactoryV1Config(**config.get("tool.web3.model_v1"))

    contract = ModelFactoryV1(model_v1_config).deploy(account, members, w3)
    config = ServerConfig(num_rounds=int(num_rounds))
    authStrategy = EthAccountStrategy(
        DecentralShapleyValueStrategy(
            fedavg_strategy,
            contract,
            coalition_to_score_fn=lambda coalition: coalition.metrics["rmse"],
            aggregate_coalition_metrics_fn=aggregate_coalitions,
        ),
        contract,
    )
    return ServerAppComponents(strategy=authStrategy, config=config)


app = ServerApp(server_fn=server_fn)
