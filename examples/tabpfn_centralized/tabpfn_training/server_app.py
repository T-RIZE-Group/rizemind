from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from tabpfn.regressor import TabPFNRegressor
from tabpfn_centralized.tabpfn_training.strategy import SimpleTabPFNRegressorStrategy
from tabpfn_centralized.tabpfn_training.task import get_weights, load_data


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    r2_scores = [num_examples * float(m["r2_score"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"weighted_average_r2_score": sum(r2_scores) / sum(examples)}


def server_fn(context: Context):
    num_rounds = int(context.run_config["num-server-rounds"])

    model_path = str(context.run_config["initial-model-path"])
    train_data, _ = load_data(partition_id=0, num_partitions=1)
    Xy_sample = train_data.sample(10)
    ndarrays = get_weights(
        TabPFNRegressor(model_path=model_path)
        .fit(X=Xy_sample.drop(["target"], axis=1), y=Xy_sample["target"])
        .model_
    )
    parameters = ndarrays_to_parameters(ndarrays)

    config = ServerConfig(num_rounds)
    strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_available_clients=int(context.run_config["min-available-clients"]),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    tabpfn_strategy = SimpleTabPFNRegressorStrategy(
        strategy=strategy, base_model_path=model_path
    )
    return ServerAppComponents(strategy=tabpfn_strategy, config=config)
