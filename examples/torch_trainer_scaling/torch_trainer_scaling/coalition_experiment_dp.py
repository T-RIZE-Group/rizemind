from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from flwr.common import ndarrays_to_parameters
from flwr.common.differential_privacy import (
    add_gaussian_noise_inplace,
    compute_stdv,
)
from flwr.server import ServerConfig
from flwr.server.strategy import DifferentialPrivacyServerSideAdaptiveClipping, FedAvg
from flwr.simulation import start_simulation

from .coalition_experiment import (
    ExperimentConfig,
    _aggregate_trainer_parameters,
    _build_client_fn,
    _build_evaluate_fn,
    _float_or_blank,
    _get_or_build_one_round_local_results,
    _history_metrics_by_round,
    _initial_model_parameters,
    build_experiment_config,
    build_parser as build_base_parser,
)
from .coalition_strategies import (
    Coalition,
    CoalitionStrategyConfig,
    build_strategy,
    coalition_to_label,
    coalition_to_mask,
    load_manual_coalitions,
)


@dataclass(frozen=True)
class DPReleaseConfig:
    noise_multiplier: float
    noise_seed: int
    initial_clipping_norm: float = 0.1
    target_clipped_quantile: float = 0.5
    mechanism: str = "flower_server_side_adaptive_clipping"


@dataclass(frozen=True)
class DPCoalitionEvaluation:
    coalition: Coalition
    accuracy: float
    loss: float
    clean_accuracy: float
    clean_loss: float
    duration_seconds: float
    execution_mode: str
    clean_duration_seconds: float
    clean_execution_mode: str
    dp_noise_std: float
    clip_norm: float
    noise_multiplier: float


def _validate_dp_config(dp_config: DPReleaseConfig) -> None:
    if dp_config.initial_clipping_norm <= 0:
        raise ValueError("--initial-clipping-norm must be positive.")
    if dp_config.noise_multiplier < 0:
        raise ValueError("--noise-multiplier must be non-negative.")
    if not 0.0 < dp_config.target_clipped_quantile <= 1.0:
        raise ValueError("--target-clipped-quantile must be in (0, 1].")


@contextmanager
def _numpy_random_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _evaluate_parameters(
    parameters: list[np.ndarray] | tuple[np.ndarray, ...],
    config: ExperimentConfig,
) -> tuple[float, float]:
    evaluate_fn = _build_evaluate_fn(config)
    loss, metrics = evaluate_fn(1, parameters, {})
    return float(loss), float(metrics["accuracy"])


def _aggregate_with_adaptive_dp(
    coalition: Coalition,
    config: ExperimentConfig,
    dp_config: DPReleaseConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float]:
    """Returns (clean_parameters, noisy_parameters, dp_noise_std, adaptive_clip_norm)."""
    initial_parameters = [
        np.array(layer, copy=True)
        for layer in _initial_model_parameters(config.seed, config.task_profile)
    ]
    if len(coalition) == 0:
        return initial_parameters, initial_parameters, 0.0, 0.0

    local_results = _get_or_build_one_round_local_results(config).results
    selected_results = [local_results[partition_id] for partition_id in coalition]
    clean_parameters = _aggregate_trainer_parameters(selected_results)

    # Compute per-client update deltas and their L2 norms
    client_deltas: list[tuple[list[np.ndarray], int]] = []
    update_norms: list[float] = []
    for result in selected_results:
        delta = [
            np.array(p, dtype=np.float64) - np.array(i, dtype=np.float64)
            for p, i in zip(result.parameters, initial_parameters)
        ]
        norm = float(np.sqrt(sum(np.sum(d * d) for d in delta)))
        update_norms.append(norm)
        client_deltas.append((delta, result.num_examples))

    # Adaptive clip norm: quantile of per-client update norms (Andrew et al.)
    adaptive_clip_norm = float(
        max(
            np.quantile(update_norms, dp_config.target_clipped_quantile),
            dp_config.initial_clipping_norm,
        )
    )

    # Per-client clipping + weighted aggregation
    total_examples = sum(n for _, n in client_deltas)
    aggregated_delta: list[np.ndarray] = [
        np.zeros_like(layer, dtype=np.float64) for layer in initial_parameters
    ]
    for delta, num_examples in client_deltas:
        norm = float(np.sqrt(sum(np.sum(d * d) for d in delta)))
        scale = min(1.0, adaptive_clip_norm / (norm + 1e-12))
        weight = num_examples / total_examples
        for i, d in enumerate(delta):
            aggregated_delta[i] += d * scale * weight

    aggregated_parameters: list[np.ndarray] = [
        np.array(init, dtype=np.float64) + agg_d
        for init, agg_d in zip(initial_parameters, aggregated_delta)
    ]

    # Noise: Flower's compute_stdv = noise_multiplier * clip_norm / num_clients
    dp_noise_std = compute_stdv(dp_config.noise_multiplier, adaptive_clip_norm, len(coalition))
    noisy_parameters = [np.array(layer, copy=True) for layer in aggregated_parameters]
    if dp_noise_std > 0.0:
        with _numpy_random_seed(dp_config.noise_seed ^ coalition_to_mask(coalition)):
            add_gaussian_noise_inplace(noisy_parameters, dp_noise_std)

    noisy_parameters = [
        np.asarray(layer, dtype=clean_parameters[layer_index].dtype)
        for layer_index, layer in enumerate(noisy_parameters)
    ]
    return clean_parameters, noisy_parameters, dp_noise_std, adaptive_clip_norm


def _evaluate_one_round_coalition_with_adaptive_dp(
    coalition: Coalition,
    config: ExperimentConfig,
    dp_config: DPReleaseConfig,
) -> DPCoalitionEvaluation:
    start = time.perf_counter()
    clean_parameters, noisy_parameters, dp_noise_std, adaptive_clip_norm = (
        _aggregate_with_adaptive_dp(coalition, config, dp_config)
    )
    clean_loss, clean_accuracy = _evaluate_parameters(clean_parameters, config)
    loss, accuracy = _evaluate_parameters(noisy_parameters, config)
    duration_seconds = time.perf_counter() - start
    return DPCoalitionEvaluation(
        coalition=coalition,
        accuracy=accuracy,
        loss=loss,
        clean_accuracy=clean_accuracy,
        clean_loss=clean_loss,
        duration_seconds=duration_seconds,
        execution_mode="flower-server-dp-adaptive-cached",
        clean_duration_seconds=duration_seconds,
        clean_execution_mode="flower-fedavg-cached",
        dp_noise_std=dp_noise_std,
        clip_norm=adaptive_clip_norm,
        noise_multiplier=dp_config.noise_multiplier,
    )


def _simulate_flower_coalition(
    coalition: Coalition,
    config: ExperimentConfig,
    *,
    dp_config: DPReleaseConfig | None,
) -> tuple[float, float, float, str]:
    if len(coalition) == 0:
        initial_parameters = _initial_model_parameters(config.seed, config.task_profile)
        loss, accuracy = _evaluate_parameters(initial_parameters, config)
        return loss, accuracy, 0.0, "flower-fedavg-empty"

    evaluate_fn = _build_evaluate_fn(config)
    base_strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=len(coalition),
        min_evaluate_clients=0,
        min_available_clients=len(coalition),
        evaluate_fn=evaluate_fn,
        initial_parameters=ndarrays_to_parameters(
            list(_initial_model_parameters(config.seed, config.task_profile))
        ),
    )
    strategy = base_strategy
    execution_mode = "flower-fedavg-simulation"
    if dp_config is not None:
        strategy = DifferentialPrivacyServerSideAdaptiveClipping(
            base_strategy,
            dp_config.noise_multiplier,
            len(coalition),
            initial_clipping_norm=dp_config.initial_clipping_norm,
            target_clipped_quantile=dp_config.target_clipped_quantile,
        )
        execution_mode = "flower-server-dp-adaptive-simulation"

    start = time.perf_counter()
    with _numpy_random_seed(
        coalition_to_mask(coalition)
        if dp_config is None
        else dp_config.noise_seed ^ coalition_to_mask(coalition)
    ):
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
    return (
        loss_by_round[last_round],
        accuracy_by_round[last_round],
        duration_seconds,
        execution_mode,
    )


def evaluate_coalitions_with_dp(
    coalitions: list[Coalition],
    config: ExperimentConfig,
    dp_config: DPReleaseConfig,
) -> list[DPCoalitionEvaluation]:
    _validate_dp_config(dp_config)
    evaluations: list[DPCoalitionEvaluation] = []

    for index, coalition in enumerate(coalitions, start=1):
        print(
            f"[{index}/{len(coalitions)}] Flower-DP evaluating coalition "
            f"{coalition_to_label(coalition)}"
        )
        if config.num_rounds == 1:
            evaluations.append(
                _evaluate_one_round_coalition_with_adaptive_dp(coalition, config, dp_config)
            )
            continue

        clean_loss, clean_accuracy, clean_duration, clean_mode = _simulate_flower_coalition(
            coalition,
            config,
            dp_config=None,
        )
        loss, accuracy, duration_seconds, execution_mode = _simulate_flower_coalition(
            coalition,
            config,
            dp_config=dp_config,
        )
        evaluations.append(
            DPCoalitionEvaluation(
                coalition=coalition,
                accuracy=float(accuracy),
                loss=float(loss),
                clean_accuracy=float(clean_accuracy),
                clean_loss=float(clean_loss),
                duration_seconds=float(duration_seconds),
                execution_mode=execution_mode,
                clean_duration_seconds=float(clean_duration),
                clean_execution_mode=clean_mode,
                dp_noise_std=compute_stdv(
                    dp_config.noise_multiplier,
                    dp_config.initial_clipping_norm,
                    len(coalition),
                ),
                clip_norm=dp_config.initial_clipping_norm,
                noise_multiplier=dp_config.noise_multiplier,
            )
        )

    return evaluations


def write_dp_config(
    output_root: Path,
    config: ExperimentConfig,
    dp_config: DPReleaseConfig,
    coalition_count: int,
) -> None:
    payload = asdict(config) | asdict(dp_config) | {"selected_coalition_count": coalition_count}
    (output_root / "config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_dp_coalitions_csv(
    output_root: Path,
    evaluations: list[DPCoalitionEvaluation],
) -> None:
    with (output_root / "coalitions.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "order",
                "coalition",
                "coalition_mask",
                "coalition_size",
                "accuracy",
                "loss",
                "clean_accuracy",
                "clean_loss",
                "duration_seconds",
                "execution_mode",
                "clean_duration_seconds",
                "clean_execution_mode",
                "dp_noise_std",
                "clip_norm",
                "noise_multiplier",
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
                    _float_or_blank(evaluation.clean_accuracy),
                    _float_or_blank(evaluation.clean_loss),
                    _float_or_blank(evaluation.duration_seconds),
                    evaluation.execution_mode,
                    _float_or_blank(evaluation.clean_duration_seconds),
                    evaluation.clean_execution_mode,
                    _float_or_blank(evaluation.dp_noise_std),
                    _float_or_blank(evaluation.clip_norm),
                    _float_or_blank(evaluation.noise_multiplier),
                ]
            )


def write_dp_metrics_by_size(
    output_root: Path,
    evaluations: list[DPCoalitionEvaluation],
) -> None:
    grouped: dict[int, list[DPCoalitionEvaluation]] = {}
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
                "mean_clean_accuracy",
                "mean_accuracy_delta",
                "mean_noise_std",
                "mean_clip_norm",
            ]
        )
        for size in sorted(grouped):
            items = grouped[size]
            writer.writerow(
                [
                    size,
                    len(items),
                    _float_or_blank(sum(item.accuracy for item in items) / len(items)),
                    _float_or_blank(sum(item.clean_accuracy for item in items) / len(items)),
                    _float_or_blank(
                        sum(item.accuracy - item.clean_accuracy for item in items) / len(items)
                    ),
                    _float_or_blank(sum(item.dp_noise_std for item in items) / len(items)),
                    _float_or_blank(sum(item.clip_norm for item in items) / len(items)),
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = (
        "Evaluate selected FL coalitions with Flower server-side adaptive-clipping DP."
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=1.0,
        help="Flower Gaussian noise multiplier. Default: 1.0",
    )
    parser.add_argument(
        "--initial-clipping-norm",
        type=float,
        default=0.1,
        help="Initial clipping norm for adaptive clipping (Andrew et al.). Default: 0.1",
    )
    parser.add_argument(
        "--target-clipped-quantile",
        type=float,
        default=0.5,
        help="Target quantile of updates to clip (0, 1]. Default: 0.5 (median).",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Optional RNG seed for Flower Gaussian noise. Defaults to --seed.",
    )
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
    dp_config = DPReleaseConfig(
        noise_multiplier=args.noise_multiplier,
        noise_seed=args.seed if args.noise_seed is None else args.noise_seed,
        initial_clipping_norm=args.initial_clipping_norm,
        target_clipped_quantile=args.target_clipped_quantile,
    )

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
        Path(config.output_dir) / f"coalition-sampling-dp-{config.strategy}-{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_dp_config(output_root, config, dp_config, len(selected_coalitions))

    print(
        f"Selected {len(selected_coalitions)} coalitions with strategy "
        f"'{config.strategy}' for Flower adaptive-DP evaluation."
    )
    evaluations = evaluate_coalitions_with_dp(selected_coalitions, config, dp_config)
    write_dp_coalitions_csv(output_root, evaluations)
    write_dp_metrics_by_size(output_root, evaluations)
    print(f"Coalition results: {(output_root / 'coalitions.csv').as_posix()}")
    print(f"Size summary: {(output_root / 'metrics_by_size.csv').as_posix()}")


if __name__ == "__main__":
    main()
