import os
import statistics
from pathlib import Path

from eth_typing import ChecksumAddress
from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import (
    Grid,
    LegacyContext,
    ServerApp,
    ServerConfig,
)
from flwr.server.strategy import FedAvg
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_strategy import EthAccountStrategy
from rizemind.configuration.toml_config import TomlConfig
from rizemind.contracts.access_control.base_access_control.base_access_control import (
    BaseAccessControlConfig,
)
from rizemind.contracts.compensation.simple_mint_compensation.simple_mint_compensation import (
    SimpleMintCompensationConfig,
)
from rizemind.contracts.contribution.contribution_calculator.contribution_calculator import (
    ContributionCalculatorConfig,
)
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import SwarmV1FactoryConfig
from rizemind.contracts.swarm.training.base_training_phase.config import (
    BaseEvaluationPhaseConfig,
    BaseTrainingPhaseConfig,
)
from rizemind.logging import LocalDiskMetricStorage, MetricStorageStrategy
from rizemind.strategies.contribution.sampling.random_deterministric import (
    RandomDeterministicSampling,
)
from rizemind.strategies.contribution.shapley.decentralized.shapley_value_strategy import (
    DecentralShapleyValueStrategy,
)
from rizemind.strategies.contribution.shapley.shapley_value_strategy import (
    TrainerSetAggregate,
)
from rizemind.swarm.config import SwarmConfig
from rizemind.swarm.modules.evaluation.assigment.swarm_task_assigner import (
    SwarmTaskAssigner,
)
from rizemind.swarm.swarm_deterministic_sampling import SwarmDeterministicSampling
from rizemind.web3.config import Web3Config
from rizemind.workflow.rizemind_workflow import RizemindWorkflow

from .task import Net, get_weights


# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def aggregate_coalitions(coalitions: list[TrainerSetAggregate]) -> dict[str, Scalar]:
    accuracies = [
        float(coalition.get_metric("accuracy", 0, aggregator=statistics.mean))
        for coalition in coalitions
    ]
    return {"median_coalition_accuracy": statistics.median(accuracies)}


app = ServerApp()


@app.lifespan()
def lifespan(context: Context):
    toml_config = TomlConfig("./pyproject.toml")
    auth_config = AccountConfig(**toml_config.get("tool.eth.account"))
    web3_config = Web3Config(**toml_config.get("tool.web3"))

    num_supernodes = int(context.run_config["num-supernodes"])
    w3 = web3_config.get_web3()
    account = auth_config.get_account(0)
    members: list[ChecksumAddress] = []
    for i in range(1, num_supernodes + 1):
        trainer = auth_config.get_account(i)
        members.append(trainer.address)

    config_dict = toml_config.get("tool.web3.swarm.factory_v1")
    config_dict.update(
        {
            "access_control": BaseAccessControlConfig(
                aggregator=account.address,
                trainers=members,
                evaluators=members,
            ),
            "compensation": SimpleMintCompensationConfig(
                token_symbol="tst",
                token_name="test",
                target_rewards=10**18,
                initial_admin=account.address,
            ),
            "contribution_calculator": ContributionCalculatorConfig(
                initial_num_samples=6,
            ),
            "training_phase": BaseTrainingPhaseConfig(ttl=35),
            "evaluation_phase": BaseEvaluationPhaseConfig(ttl=30, registration_ttl=10),
        }
    )
    swarm_factory_config = SwarmV1FactoryConfig(
        **config_dict,
    )
    swarm_config = SwarmConfig(
        address=toml_config.get("tool.web3.swarm.address"),
        factory_v1=swarm_factory_config,
    )
    swarm = swarm_config.get_or_deploy(deployer=account, w3=w3)
    # Hacky way to make the swarm address available on client startup
    # TomlConfig will parse the environment variable
    os.environ["DEPLOYED_SWARM_ADDRESS"] = swarm.address
    swarm_config.address = swarm.address
    web3_config.store_in_context(context)
    swarm_config.store_in_context(context)
    auth_config.store_in_context(context)
    yield
    del os.environ["DEPLOYED_SWARM_ADDRESS"]


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_available_clients=int(context.run_config["min-available-clients"]),
        min_fit_clients=int(context.run_config["min-available-clients"]),
        min_evaluate_clients=int(context.run_config["min-available-clients"]),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    server_config = ServerConfig(num_rounds=num_rounds)

    auth_config = AccountConfig.from_context(context)
    web3_config = Web3Config.from_context(context)
    swarm_config = SwarmConfig.from_context(context)

    if swarm_config is None or auth_config is None or web3_config is None:
        raise ValueError("Missing config in context")

    w3 = web3_config.get_web3()
    account = auth_config.get_account(0)
    swarm = swarm_config.get(account=account, w3=w3)
    authStrategy = EthAccountStrategy(
        DecentralShapleyValueStrategy(
            strategy,
            swarm,
            coalition_to_score_fn=lambda coalition: coalition.get_metric(
                "accuracy", 0, aggregator=statistics.mean
            ),
            aggregate_coalition_metrics_fn=aggregate_coalitions,
            shapley_sampling_strat=RandomDeterministicSampling(
                SwarmDeterministicSampling(swarm)
            ),
            task_assigner=SwarmTaskAssigner(swarm),
        ),
        swarm,
        account,
    )
    toml_config = TomlConfig("./pyproject.toml")
    metrics_storage = LocalDiskMetricStorage(
        Path(str(context.run_config["metrics-storage-path"])),
        "torch-shapley",
    )
    metrics_storage.write_config(context.run_config)
    metrics_storage.write_config(toml_config.data)
    metrics_strategy = MetricStorageStrategy(authStrategy, metrics_storage)
    workflow = RizemindWorkflow(swarm=swarm)
    context = LegacyContext(
        context=context,
        config=server_config,
        strategy=metrics_strategy,
    )
    workflow(grid, context)
