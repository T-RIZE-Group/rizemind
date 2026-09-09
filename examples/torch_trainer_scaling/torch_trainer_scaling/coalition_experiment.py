from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.history import History
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from .coalition_strategies import (
    STRATEGY_REGISTRY,
    Coalition,
    CoalitionStrategyConfig,
    build_strategy,
    coalition_to_label,
    coalition_to_mask,
    load_manual_coalitions,
)
from .runtime import resolve_device
from .task import (
    DatasetName,
    TaskProfile,
    build_model,
    get_testloader,
    get_trainloader,
    get_weights,
    set_weights,
    test,
    train,
)


def _float_or_blank(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _history_metrics_by_round(
    history: History,
) -> tuple[dict[int, float], dict[int, float]]:
    accuracy_by_round = {
        int(server_round): float(value)
        for server_round, value in history.metrics_centralized.get("accuracy", [])
    }
    loss_by_round = {
        int(server_round): float(loss)
        for server_round, loss in history.losses_centralized
    }
    return accuracy_by_round, loss_by_round


@dataclass(frozen=True)
class CoalitionEvaluation:
    coalition: Coalition
    accuracy: float
    loss: float
    duration_seconds: float
    execution_mode: str
    aggregation_duration_seconds: float | None = None
    evaluation_duration_seconds: float | None = None


@dataclass(frozen=True)
class TrainerLocalResult:
    partition_id: int
    parameters: tuple[np.ndarray, ...]
    num_examples: int
    train_loss: float


@dataclass(frozen=True)
class OneRoundLocalResultCacheEntry:
    results: dict[int, TrainerLocalResult]
    build_duration_seconds: float


@dataclass(frozen=True)
class EvaluationContext:
    testloader: object
    device_name: str


@dataclass(frozen=True)
class ExperimentConfig:
    total_trainers: int
    strategy: str
    num_rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    dataset: DatasetName
    task_profile: TaskProfile
    data_dir: str
    output_dir: str
    max_train_samples_per_trainer: int | None
    max_test_samples: int | None
    seed: int
    budget: int | None
    budget_per_size: int | None
    min_size: int
    max_size: int | None
    coalitions: str | None
    coalitions_file: str | None
    include_empty: bool
    include_grand: bool
    device: str
    client_num_cpus: float
    client_num_gpus: float
    train_loader_workers: int
    test_loader_workers: int
    persistent_workers: bool
    evaluation_workers: int
    torch_threads_per_worker: int
    evaluation_device: str | None


ProgressCallback = Callable[[int, int, Coalition], None]


_ONE_ROUND_LOCAL_RESULT_CACHE: dict[
    tuple[
        int,
        int,
        int,
        float,
        DatasetName,
        TaskProfile,
        str,
        int | None,
        int | None,
        int,
        str,
        int,
        bool,
    ],
    OneRoundLocalResultCacheEntry,
] = {}

_EVALUATION_CONTEXT_CACHE: dict[
    tuple[
        int,
        int,
        DatasetName,
        TaskProfile,
        str,
        int | None,
        int | None,
        int,
        str,
        int,
        bool,
    ],
    EvaluationContext,
] = {}

_INITIAL_MODEL_PARAMETERS: dict[tuple[int, TaskProfile], tuple[np.ndarray, ...]] = {}
_PARALLEL_LOCAL_RESULTS: dict[int, TrainerLocalResult] | None = None
_PARALLEL_EVALUATE_FN: Callable | None = None
_PARALLEL_INITIAL_PARAMETERS: tuple[np.ndarray, ...] | None = None


class CoalitionUtilityClient(NumPyClient):
    def __init__(
        self,
        global_partition_id: int,
        total_trainers: int,
        batch_size: int,
        local_epochs: int,
        learning_rate: float,
        dataset_name: DatasetName,
        task_profile: TaskProfile,
        data_dir: str,
        max_train_samples_per_trainer: int | None,
        max_test_samples: int | None,
        seed: int,
        device_name: str,
        train_loader_workers: int,
        test_loader_workers: int,
        persistent_workers: bool,
    ) -> None:
        self.global_partition_id = global_partition_id
        self.total_trainers = total_trainers
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
        self.task_profile = task_profile
        self.data_dir = data_dir
        self.max_train_samples_per_trainer = max_train_samples_per_trainer
        self.max_test_samples = max_test_samples
        self.seed = seed
        self.device = resolve_device(device_name)
        self.train_loader_workers = train_loader_workers
        self.test_loader_workers = test_loader_workers
        self.persistent_workers = persistent_workers

    def fit(self, parameters, config):
        trainloader = get_trainloader(
            partition_id=self.global_partition_id,
            batch_size=self.batch_size,
            dataset_name=self.dataset_name,
            task_profile=self.task_profile,
            data_dir=self.data_dir,
            num_trainers=self.total_trainers,
            max_train_samples_per_trainer=self.max_train_samples_per_trainer,
            max_test_samples=self.max_test_samples,
            seed=self.seed,
            num_workers=self.train_loader_workers,
            persistent_workers=self.persistent_workers,
        )
        model = build_model(self.task_profile)
        set_weights(model, parameters)
        train_loss = train(
            net=model,
            trainloader=trainloader,
            epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            device=self.device,
        )
        return get_weights(model), len(trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        testloader = get_testloader(
            batch_size=self.batch_size,
            dataset_name=self.dataset_name,
            task_profile=self.task_profile,
            data_dir=self.data_dir,
            num_trainers=self.total_trainers,
            max_train_samples_per_trainer=self.max_train_samples_per_trainer,
            max_test_samples=self.max_test_samples,
            seed=self.seed,
            num_workers=self.test_loader_workers,
            persistent_workers=self.persistent_workers,
        )
        model = build_model(self.task_profile)
        set_weights(model, parameters)
        loss, accuracy = test(model, testloader, self.device)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def _build_client_fn(
    coalition: Coalition,
    config: ExperimentConfig,
):
    def client_fn(context: Context):
        local_partition_id = int(context.node_config["partition-id"])
        global_partition_id = coalition[local_partition_id]
        return CoalitionUtilityClient(
            global_partition_id=global_partition_id,
            total_trainers=config.total_trainers,
            batch_size=config.batch_size,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            dataset_name=config.dataset,
            task_profile=config.task_profile,
            data_dir=config.data_dir,
            max_train_samples_per_trainer=config.max_train_samples_per_trainer,
            max_test_samples=config.max_test_samples,
            seed=config.seed,
            device_name=config.device,
            train_loader_workers=config.train_loader_workers,
            test_loader_workers=config.test_loader_workers,
            persistent_workers=config.persistent_workers,
        ).to_client()

    return client_fn


def _resolved_evaluation_device_name(config: ExperimentConfig) -> str:
    return config.device if config.evaluation_device is None else config.evaluation_device


def _evaluation_context_cache_key(
    config: ExperimentConfig,
) -> tuple[
    int,
    int,
    DatasetName,
    TaskProfile,
    str,
    int | None,
    int | None,
    int,
    str,
    int,
    bool,
]:
    return (
        config.total_trainers,
        config.batch_size,
        config.dataset,
        config.task_profile,
        config.data_dir,
        config.max_train_samples_per_trainer,
        config.max_test_samples,
        config.seed,
        _resolved_evaluation_device_name(config),
        config.test_loader_workers,
        config.persistent_workers,
    )


def _get_or_build_evaluation_context(config: ExperimentConfig) -> EvaluationContext:
    cache_key = _evaluation_context_cache_key(config)
    cached_context = _EVALUATION_CONTEXT_CACHE.get(cache_key)
    if cached_context is not None:
        return cached_context

    testloader = get_testloader(
        batch_size=config.batch_size,
        dataset_name=config.dataset,
        task_profile=config.task_profile,
        data_dir=config.data_dir,
        num_trainers=config.total_trainers,
        max_train_samples_per_trainer=config.max_train_samples_per_trainer,
        max_test_samples=config.max_test_samples,
        seed=config.seed,
        num_workers=config.test_loader_workers,
        persistent_workers=config.persistent_workers,
    )
    context = EvaluationContext(
        testloader=testloader,
        device_name=_resolved_evaluation_device_name(config),
    )
    _EVALUATION_CONTEXT_CACHE[cache_key] = context
    return context


def _build_evaluate_fn(config: ExperimentConfig):
    context = _get_or_build_evaluation_context(config)
    device = resolve_device(context.device_name)

    def evaluate_fn(server_round: int, parameters_ndarrays, evaluate_config):
        model = build_model(config.task_profile)
        set_weights(model, parameters_ndarrays)
        loss, accuracy = test(model, context.testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn


def _one_round_cache_key(
    config: ExperimentConfig,
) -> tuple[
    int,
    int,
    int,
    float,
    DatasetName,
    TaskProfile,
    str,
    int | None,
    int | None,
    int,
    str,
    int,
    bool,
]:
    return (
        config.total_trainers,
        config.batch_size,
        config.local_epochs,
        config.learning_rate,
        config.dataset,
        config.task_profile,
        config.data_dir,
        config.max_train_samples_per_trainer,
        config.max_test_samples,
        config.seed,
        config.device,
        config.train_loader_workers,
        config.persistent_workers,
    )


def _initial_model_parameters(
    seed: int, task_profile: TaskProfile
) -> tuple[np.ndarray, ...]:
    cache_key = (seed, task_profile)
    cached_parameters = _INITIAL_MODEL_PARAMETERS.get(cache_key)
    if cached_parameters is None:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            cached_parameters = tuple(
                np.array(layer, copy=True)
                for layer in get_weights(build_model(task_profile))
            )
        _INITIAL_MODEL_PARAMETERS[cache_key] = cached_parameters
    return tuple(np.array(layer, copy=True) for layer in cached_parameters)


def _train_single_trainer(
    partition_id: int,
    config: ExperimentConfig,
    *,
    initial_parameters: tuple[np.ndarray, ...],
) -> TrainerLocalResult:
    trainloader = get_trainloader(
        partition_id=partition_id,
        batch_size=config.batch_size,
        dataset_name=config.dataset,
        task_profile=config.task_profile,
        data_dir=config.data_dir,
        num_trainers=config.total_trainers,
        max_train_samples_per_trainer=config.max_train_samples_per_trainer,
        max_test_samples=config.max_test_samples,
        seed=config.seed,
        num_workers=config.train_loader_workers,
        persistent_workers=config.persistent_workers,
    )
    model = build_model(config.task_profile)
    set_weights(model, initial_parameters)
    train_loss = train(
        net=model,
        trainloader=trainloader,
        epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        device=resolve_device(config.device),
    )
    return TrainerLocalResult(
        partition_id=partition_id,
        parameters=tuple(np.array(layer, copy=True) for layer in get_weights(model)),
        num_examples=len(trainloader.dataset),
        train_loss=train_loss,
    )


def _get_or_build_one_round_local_results(
    config: ExperimentConfig,
) -> OneRoundLocalResultCacheEntry:
    cache_key = _one_round_cache_key(config)
    cached_results = _ONE_ROUND_LOCAL_RESULT_CACHE.get(cache_key)
    if cached_results is not None:
        return cached_results

    print(
        "Pretraining local models once for "
        f"{config.total_trainers} trainers (num_rounds=1)."
    )
    build_start = time.perf_counter()
    initial_parameters = _initial_model_parameters(config.seed, config.task_profile)
    local_results = {
        partition_id: _train_single_trainer(
            partition_id,
            config,
            initial_parameters=initial_parameters,
        )
        for partition_id in range(config.total_trainers)
    }
    cache_entry = OneRoundLocalResultCacheEntry(
        results=local_results,
        build_duration_seconds=time.perf_counter() - build_start,
    )
    _ONE_ROUND_LOCAL_RESULT_CACHE[cache_key] = cache_entry
    return cache_entry


def _aggregate_trainer_parameters(
    local_results: Sequence[TrainerLocalResult],
) -> list[np.ndarray]:
    total_examples = sum(result.num_examples for result in local_results)
    if total_examples <= 0:
        raise ValueError("Cannot aggregate trainer models with zero total examples.")

    aggregated: list[np.ndarray] = []
    for layer_index in range(len(local_results[0].parameters)):
        weighted_sum = np.zeros_like(local_results[0].parameters[layer_index])
        for result in local_results:
            weighted_sum += result.parameters[layer_index] * result.num_examples
        aggregated.append(weighted_sum / total_examples)
    return aggregated


def _execution_mode_label(config: ExperimentConfig) -> str:
    if config.num_rounds != 1:
        return "simulation"

    evaluation_device = resolve_device(_resolved_evaluation_device_name(config)).type
    if config.evaluation_workers > 1 and evaluation_device == "cpu":
        return "parallel-cpu-eval"
    if evaluation_device == "mps":
        return "serial-mps"
    return "serial-cpu"


def _set_worker_torch_threads(torch_threads_per_worker: int) -> None:
    threads = max(torch_threads_per_worker, 1)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _init_parallel_evaluation_worker(
    config: ExperimentConfig,
    local_results: dict[int, TrainerLocalResult],
    torch_threads_per_worker: int,
) -> None:
    global _PARALLEL_EVALUATE_FN, _PARALLEL_INITIAL_PARAMETERS, _PARALLEL_LOCAL_RESULTS

    _set_worker_torch_threads(torch_threads_per_worker)
    _PARALLEL_LOCAL_RESULTS = local_results
    _PARALLEL_EVALUATE_FN = _build_evaluate_fn(config)
    _PARALLEL_INITIAL_PARAMETERS = _initial_model_parameters(
        config.seed, config.task_profile
    )


def _evaluate_one_round_coalition_worker(
    coalition: Coalition,
) -> CoalitionEvaluation:
    if _PARALLEL_EVALUATE_FN is None:
        raise RuntimeError("Parallel evaluation worker is not initialized.")
    if _PARALLEL_INITIAL_PARAMETERS is None:
        raise RuntimeError("Parallel initial-parameter cache is not initialized.")
    if _PARALLEL_LOCAL_RESULTS is None:
        raise RuntimeError("Parallel local-result cache is not initialized.")

    return _evaluate_cached_one_round_coalition(
        coalition=coalition,
        local_results_by_partition=_PARALLEL_LOCAL_RESULTS,
        evaluate_fn=_PARALLEL_EVALUATE_FN,
        execution_mode="parallel-cpu-eval",
        initial_parameters=_PARALLEL_INITIAL_PARAMETERS,
    )


def _evaluate_cached_one_round_coalition(
    coalition: Coalition,
    *,
    local_results_by_partition: dict[int, TrainerLocalResult],
    evaluate_fn: Callable,
    execution_mode: str,
    initial_parameters: tuple[np.ndarray, ...],
) -> CoalitionEvaluation:
    aggregation_start = time.perf_counter()
    if len(coalition) == 0:
        aggregated_parameters = initial_parameters
    else:
        selected_results = [
            local_results_by_partition[partition_id] for partition_id in coalition
        ]
        aggregated_parameters = _aggregate_trainer_parameters(selected_results)
    aggregation_duration_seconds = time.perf_counter() - aggregation_start

    evaluation_start = time.perf_counter()
    loss, metrics = evaluate_fn(1, aggregated_parameters, {})
    evaluation_duration_seconds = time.perf_counter() - evaluation_start
    return CoalitionEvaluation(
        coalition=coalition,
        accuracy=float(metrics["accuracy"]),
        loss=float(loss),
        duration_seconds=aggregation_duration_seconds + evaluation_duration_seconds,
        execution_mode=execution_mode,
        aggregation_duration_seconds=aggregation_duration_seconds,
        evaluation_duration_seconds=evaluation_duration_seconds,
    )


def _evaluate_empty_coalition(config: ExperimentConfig) -> CoalitionEvaluation:
    evaluate_fn = _build_evaluate_fn(config)
    return _evaluate_cached_one_round_coalition(
        coalition=(),
        local_results_by_partition={},
        evaluate_fn=evaluate_fn,
        execution_mode=_execution_mode_label(config),
        initial_parameters=_initial_model_parameters(config.seed, config.task_profile),
    )


def _evaluate_trained_coalition(
    coalition: Coalition,
    config: ExperimentConfig,
) -> CoalitionEvaluation:
    if config.num_rounds == 1:
        return _evaluate_one_round_coalition(coalition, config)

    evaluate_fn = _build_evaluate_fn(config)
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=len(coalition),
        min_evaluate_clients=0,
        min_available_clients=len(coalition),
        evaluate_fn=evaluate_fn,
        initial_parameters=ndarrays_to_parameters(
            get_weights(build_model(config.task_profile))
        ),
    )

    start = time.perf_counter()
    history = start_simulation(
        client_fn=_build_client_fn(coalition, config),
        num_clients=len(coalition),
        config=ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": config.client_num_cpus,
            "num_gpus": config.client_num_gpus,
        },
        ray_init_args={
            "address": "local",
            "ignore_reinit_error": True,
            "include_dashboard": False,
        },
    )
    duration_seconds = time.perf_counter() - start
    accuracy_by_round, loss_by_round = _history_metrics_by_round(history)
    last_round = max(set(accuracy_by_round) | set(loss_by_round))

    return CoalitionEvaluation(
        coalition=coalition,
        accuracy=accuracy_by_round[last_round],
        loss=loss_by_round[last_round],
        duration_seconds=duration_seconds,
        execution_mode="simulation",
    )


def _evaluate_one_round_coalition(
    coalition: Coalition,
    config: ExperimentConfig,
) -> CoalitionEvaluation:
    evaluate_fn = _build_evaluate_fn(config)
    cache_entry = _get_or_build_one_round_local_results(config)
    return _evaluate_cached_one_round_coalition(
        coalition=coalition,
        local_results_by_partition=cache_entry.results,
        evaluate_fn=evaluate_fn,
        execution_mode=_execution_mode_label(config),
        initial_parameters=_initial_model_parameters(config.seed, config.task_profile),
    )


def evaluate_coalition(
    coalition: Coalition,
    config: ExperimentConfig,
) -> CoalitionEvaluation:
    if len(coalition) == 0:
        return _evaluate_empty_coalition(config)
    return _evaluate_trained_coalition(coalition, config)


def evaluate_coalitions(
    coalitions: Sequence[Coalition],
    config: ExperimentConfig,
    *,
    cache: dict[Coalition, CoalitionEvaluation] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[CoalitionEvaluation]:
    evaluation_cache = {} if cache is None else cache
    total = len(coalitions)

    if (
        config.num_rounds == 1
        and config.evaluation_workers > 1
        and resolve_device(_resolved_evaluation_device_name(config)).type == "cpu"
    ):
        print(
            "One-round execution mode: parallel-cpu-eval "
            f"(workers={config.evaluation_workers}, "
            f"torch_threads_per_worker={config.torch_threads_per_worker}, "
            f"test_loader_workers={config.test_loader_workers})."
        )
        cache_entry = _get_or_build_one_round_local_results(config)
        missing_coalitions = [
            coalition
            for coalition in coalitions
            if coalition not in evaluation_cache
        ]
        if missing_coalitions:
            with ProcessPoolExecutor(
                max_workers=config.evaluation_workers,
                initializer=_init_parallel_evaluation_worker,
                initargs=(
                    config,
                    cache_entry.results,
                    config.torch_threads_per_worker,
                ),
            ) as executor:
                future_by_coalition = {
                    executor.submit(_evaluate_one_round_coalition_worker, coalition): coalition
                    for coalition in missing_coalitions
                }
                completed = 0
                for future in as_completed(future_by_coalition):
                    evaluation = future.result()
                    completed += 1
                    evaluation_cache[evaluation.coalition] = evaluation
                    if progress_callback is not None:
                        progress_callback(completed, len(missing_coalitions), evaluation.coalition)
        return [evaluation_cache[coalition] for coalition in coalitions]

    if config.num_rounds == 1:
        print(
            "One-round execution mode: "
            f"{_execution_mode_label(config)} "
            f"(test_loader_workers={config.test_loader_workers})."
        )

    evaluations: list[CoalitionEvaluation] = []

    for index, coalition in enumerate(coalitions, start=1):
        if coalition not in evaluation_cache:
            if progress_callback is not None:
                progress_callback(index, total, coalition)
            evaluation_cache[coalition] = evaluate_coalition(coalition, config)
        evaluations.append(evaluation_cache[coalition])

    return evaluations


def get_one_round_pretraining_duration(config: ExperimentConfig) -> float:
    if config.num_rounds != 1:
        return 0.0
    cache_entry = _ONE_ROUND_LOCAL_RESULT_CACHE.get(_one_round_cache_key(config))
    if cache_entry is None:
        return 0.0
    return cache_entry.build_duration_seconds


def has_one_round_local_results(config: ExperimentConfig) -> bool:
    if config.num_rounds != 1:
        return False
    return _one_round_cache_key(config) in _ONE_ROUND_LOCAL_RESULT_CACHE


def get_execution_mode(config: ExperimentConfig) -> str:
    return _execution_mode_label(config)


def write_config(
    output_root: Path, config: ExperimentConfig, coalition_count: int
) -> None:
    payload = asdict(config) | {"selected_coalition_count": coalition_count}
    (output_root / "config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_coalitions_csv(
    output_root: Path,
    evaluations: Sequence[CoalitionEvaluation],
) -> None:
    with (output_root / "coalitions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "order",
                "coalition",
                "coalition_mask",
                "coalition_size",
                "accuracy",
                "loss",
                "duration_seconds",
                "execution_mode",
                "aggregation_duration_seconds",
                "evaluation_duration_seconds",
            ]
        )
        for order, evaluation in enumerate(evaluations):
            writer.writerow(
                [
                    order,
                    coalition_to_label(evaluation.coalition),
                    coalition_to_mask(evaluation.coalition),
                    len(evaluation.coalition),
                    _float_or_blank(evaluation.accuracy),
                    _float_or_blank(evaluation.loss),
                    _float_or_blank(evaluation.duration_seconds),
                    evaluation.execution_mode,
                    _float_or_blank(evaluation.aggregation_duration_seconds),
                    _float_or_blank(evaluation.evaluation_duration_seconds),
                ]
            )


def write_metrics_by_size(
    output_root: Path,
    evaluations: Sequence[CoalitionEvaluation],
) -> None:
    grouped: dict[int, list[CoalitionEvaluation]] = {}
    for evaluation in evaluations:
        grouped.setdefault(len(evaluation.coalition), []).append(evaluation)

    with (output_root / "metrics_by_size.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "coalition_size",
                "num_coalitions",
                "mean_accuracy",
                "median_accuracy",
                "best_accuracy",
                "mean_loss",
            ]
        )
        for size in sorted(grouped):
            items = grouped[size]
            accuracies = [item.accuracy for item in items]
            losses = [item.loss for item in items]
            writer.writerow(
                [
                    size,
                    len(items),
                    _float_or_blank(statistics.mean(accuracies)),
                    _float_or_blank(statistics.median(accuracies)),
                    _float_or_blank(max(accuracies)),
                    _float_or_blank(statistics.mean(losses)),
                ]
            )


def add_shared_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Federated rounds per coalition evaluation. Default: 1",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local epochs per selected trainer. Default: 1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and centralized evaluation. Default: 32",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for local optimization. Default: 0.001",
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fake"],
        default="mnist",
        help="Dataset to use. 'fake' is useful for smoke tests.",
    )
    parser.add_argument(
        "--task-profile",
        choices=["repo", "flower_fl_dp_sa"],
        default="repo",
        help="Task/data profile. 'flower_fl_dp_sa' mirrors Flower's fl-dp-sa tutorial defaults for MNIST partitioning, normalization, and model.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory used for dataset downloads/cache. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Directory where coalition experiment logs are written. Default: logs",
    )
    parser.add_argument(
        "--max-train-samples-per-trainer",
        type=int,
        default=None,
        help="Optional cap on train samples per trainer to speed up experiments.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on centralized test samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for coalition sampling and partitioning. Default: 42",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Training/evaluation device. Default: auto",
    )
    parser.add_argument(
        "--client-num-cpus",
        type=float,
        default=1.0,
        help="CPU resources reserved per simulated client. Default: 1.0",
    )
    parser.add_argument(
        "--client-num-gpus",
        type=float,
        default=0.0,
        help="GPU resources reserved per simulated client. Default: 0.0",
    )
    parser.add_argument(
        "--train-loader-workers",
        type=int,
        default=0,
        help="Worker processes for train DataLoader. Default: 0",
    )
    parser.add_argument(
        "--test-loader-workers",
        type=int,
        default=0,
        help="Worker processes for test DataLoader. Default: 0",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep DataLoader workers alive between iterations when workers > 0. Default: false",
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=1,
        help="Process workers for one-round CPU coalition evaluation. Default: 1",
    )
    parser.add_argument(
        "--torch-threads-per-worker",
        type=int,
        default=1,
        help="PyTorch intra-op threads per evaluation worker. Default: 1",
    )
    parser.add_argument(
        "--evaluation-device",
        choices=["auto", "cpu", "mps", "cuda"],
        default=None,
        help="Optional device override for centralized evaluation. Defaults to --device.",
    )


def build_experiment_config(
    args: argparse.Namespace,
    *,
    total_trainers: int | None = None,
    strategy: str | None = None,
) -> ExperimentConfig:
    if args.train_loader_workers < 0 or args.test_loader_workers < 0:
        raise ValueError("DataLoader worker counts must be non-negative.")
    if args.evaluation_workers <= 0:
        raise ValueError("--evaluation-workers must be positive.")
    if args.torch_threads_per_worker <= 0:
        raise ValueError("--torch-threads-per-worker must be positive.")

    resolved_total_trainers = (
        args.total_trainers if total_trainers is None else total_trainers
    )
    resolved_strategy = args.strategy if strategy is None else strategy
    return ExperimentConfig(
        total_trainers=resolved_total_trainers,
        strategy=resolved_strategy,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dataset=args.dataset,
        task_profile=args.task_profile,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_train_samples_per_trainer=args.max_train_samples_per_trainer,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        budget=getattr(args, "budget", None),
        budget_per_size=getattr(args, "budget_per_size", None),
        min_size=getattr(args, "min_size", 0),
        max_size=getattr(args, "max_size", None),
        coalitions=getattr(args, "coalitions", None),
        coalitions_file=getattr(args, "coalitions_file", None),
        include_empty=getattr(args, "include_empty", True),
        include_grand=getattr(args, "include_grand", True),
        device=getattr(args, "device", "auto"),
        client_num_cpus=getattr(args, "client_num_cpus", 1.0),
        client_num_gpus=getattr(args, "client_num_gpus", 0.0),
        train_loader_workers=getattr(args, "train_loader_workers", 0),
        test_loader_workers=getattr(args, "test_loader_workers", 0),
        persistent_workers=getattr(args, "persistent_workers", False),
        evaluation_workers=getattr(args, "evaluation_workers", 1),
        torch_threads_per_worker=getattr(args, "torch_threads_per_worker", 1),
        evaluation_device=getattr(args, "evaluation_device", None),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate selected FL coalitions using pluggable coalition feeder strategies."
    )
    parser.add_argument(
        "--total-trainers",
        type=int,
        default=10,
        help="Total number of fixed trainer partitions. Default: 10",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_REGISTRY),
        default="stratified",
        help="Coalition feeder strategy to use. Default: stratified",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Total coalition budget for the uniform strategy.",
    )
    parser.add_argument(
        "--budget-per-size",
        type=int,
        default=1,
        help="Coalitions to sample per coalition size for the stratified strategy. Default: 1",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        help="Minimum coalition size for stratified sampling. Default: 0",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum coalition size for stratified sampling. Default: total trainers",
    )
    parser.add_argument(
        "--coalitions",
        default=None,
        help='Manual coalitions as semicolon-separated trainer lists, for example "empty;0;0,1,2;4,7".',
    )
    parser.add_argument(
        "--coalitions-file",
        default=None,
        help="Path to a text file with one manual coalition per line.",
    )
    parser.add_argument(
        "--include-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the empty coalition. Default: true",
    )
    parser.add_argument(
        "--include-grand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the grand coalition. Default: true",
    )
    add_shared_experiment_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manual_coalitions = tuple(
        load_manual_coalitions(
            raw_value=args.coalitions,
            file_path=args.coalitions_file,
            total_trainers=args.total_trainers,
        )
    )

    config = build_experiment_config(args)

    strategy = build_strategy(
        CoalitionStrategyConfig(
            strategy=config.strategy,
            total_trainers=config.total_trainers,
            seed=config.seed,
            budget=config.budget,
            budget_per_size=config.budget_per_size,
            min_size=config.min_size,
            max_size=config.max_size,
            include_empty=config.include_empty,
            include_grand=config.include_grand,
            manual_coalitions=manual_coalitions,
        )
    )
    selected_coalitions = strategy.generate()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = (
        Path(config.output_dir) / f"coalition-sampling-{config.strategy}-{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_config(output_root, config, len(selected_coalitions))

    print(
        f"Selected {len(selected_coalitions)} coalitions "
        f"with strategy '{config.strategy}'."
    )
    evaluations = evaluate_coalitions(
        selected_coalitions,
        config,
        progress_callback=lambda index, total, coalition: print(
            f"[{index}/{total}] Evaluating coalition "
            f"{coalition_to_label(coalition)}"
        ),
    )

    write_coalitions_csv(output_root, evaluations)
    write_metrics_by_size(output_root, evaluations)
    print(f"Coalition results: {(output_root / 'coalitions.csv').as_posix()}")
    print(f"Size summary: {(output_root / 'metrics_by_size.csv').as_posix()}")


if __name__ == "__main__":
    main()
