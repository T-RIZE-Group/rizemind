from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flwr.client import NumPyClient
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.history import History
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from .runtime import resolve_device
from .task import (
    DatasetName,
    Net,
    get_testloader,
    get_trainloader,
    get_weights,
    set_weights,
    test,
    train,
)


def _parse_trainer_counts(raw_value: str) -> list[int]:
    counts: list[int] = []
    for value in raw_value.split(","):
        cleaned = value.strip()
        if not cleaned:
            continue
        count = int(cleaned)
        if count <= 0:
            raise ValueError("Trainer counts must be positive integers.")
        counts.append(count)
    if not counts:
        raise ValueError("Provide at least one trainer count.")
    return counts


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


@dataclass
class ExperimentResult:
    trainer_count: int
    final_accuracy: float | None
    best_accuracy: float | None
    final_loss: float | None
    run_dir: Path


class TrainerScalingClient(NumPyClient):
    def __init__(
        self,
        partition_id: int,
        num_trainers: int,
        batch_size: int,
        local_epochs: int,
        learning_rate: float,
        dataset_name: DatasetName,
        data_dir: str,
        max_train_samples_per_trainer: int | None,
        max_test_samples: int | None,
        seed: int,
        device_name: str,
        train_loader_workers: int,
        test_loader_workers: int,
        persistent_workers: bool,
    ) -> None:
        self.partition_id = partition_id
        self.num_trainers = num_trainers
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
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
            partition_id=self.partition_id,
            batch_size=self.batch_size,
            dataset_name=self.dataset_name,
            data_dir=self.data_dir,
            num_trainers=self.num_trainers,
            max_train_samples_per_trainer=self.max_train_samples_per_trainer,
            max_test_samples=self.max_test_samples,
            seed=self.seed,
            num_workers=self.train_loader_workers,
            persistent_workers=self.persistent_workers,
        )
        model = Net()
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
            data_dir=self.data_dir,
            num_trainers=self.num_trainers,
            max_train_samples_per_trainer=self.max_train_samples_per_trainer,
            max_test_samples=self.max_test_samples,
            seed=self.seed,
            num_workers=self.test_loader_workers,
            persistent_workers=self.persistent_workers,
        )
        model = Net()
        set_weights(model, parameters)
        loss, accuracy = test(model, testloader, self.device)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def _build_client_fn(
    trainer_count: int,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    dataset_name: DatasetName,
    data_dir: str,
    max_train_samples_per_trainer: int | None,
    max_test_samples: int | None,
    seed: int,
    device_name: str,
    train_loader_workers: int,
    test_loader_workers: int,
    persistent_workers: bool,
):
    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        return TrainerScalingClient(
            partition_id=partition_id,
            num_trainers=trainer_count,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            dataset_name=dataset_name,
            data_dir=data_dir,
            max_train_samples_per_trainer=max_train_samples_per_trainer,
            max_test_samples=max_test_samples,
            seed=seed,
            device_name=device_name,
            train_loader_workers=train_loader_workers,
            test_loader_workers=test_loader_workers,
            persistent_workers=persistent_workers,
        ).to_client()

    return client_fn


def _build_evaluate_fn(
    trainer_count: int,
    batch_size: int,
    dataset_name: DatasetName,
    data_dir: str,
    max_train_samples_per_trainer: int | None,
    max_test_samples: int | None,
    seed: int,
    device_name: str,
    test_loader_workers: int,
    persistent_workers: bool,
):
    testloader = get_testloader(
        batch_size=batch_size,
        dataset_name=dataset_name,
        data_dir=data_dir,
        num_trainers=trainer_count,
        max_train_samples_per_trainer=max_train_samples_per_trainer,
        max_test_samples=max_test_samples,
        seed=seed,
        num_workers=test_loader_workers,
        persistent_workers=persistent_workers,
    )
    device = resolve_device(device_name)

    def evaluate_fn(server_round: int, parameters_ndarrays, config):
        model = Net()
        set_weights(model, parameters_ndarrays)
        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn


def _write_round_metrics(
    history: History, destination: Path
) -> tuple[float | None, float | None, float | None]:
    accuracy_by_round, loss_by_round = _history_metrics_by_round(history)
    rounds = sorted(set(accuracy_by_round) | set(loss_by_round))

    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["server_round", "accuracy", "loss"])
        for server_round in rounds:
            writer.writerow(
                [
                    server_round,
                    _float_or_blank(accuracy_by_round.get(server_round)),
                    _float_or_blank(loss_by_round.get(server_round)),
                ]
            )

    final_accuracy = accuracy_by_round.get(rounds[-1]) if rounds else None
    best_accuracy = max(accuracy_by_round.values()) if accuracy_by_round else None
    final_loss = loss_by_round.get(rounds[-1]) if rounds else None
    return final_accuracy, best_accuracy, final_loss


def _run_single_experiment(
    trainer_count: int,
    num_rounds: int,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    dataset_name: DatasetName,
    data_dir: str,
    max_train_samples_per_trainer: int | None,
    max_test_samples: int | None,
    seed: int,
    device_name: str,
    client_num_cpus: float,
    client_num_gpus: float,
    train_loader_workers: int,
    test_loader_workers: int,
    persistent_workers: bool,
    output_root: Path,
) -> ExperimentResult:
    run_dir = output_root / f"trainers-{trainer_count}"
    run_dir.mkdir(parents=True, exist_ok=True)

    client_fn = _build_client_fn(
        trainer_count=trainer_count,
        batch_size=batch_size,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        dataset_name=dataset_name,
        data_dir=data_dir,
        max_train_samples_per_trainer=max_train_samples_per_trainer,
        max_test_samples=max_test_samples,
        seed=seed,
        device_name=device_name,
        train_loader_workers=train_loader_workers,
        test_loader_workers=test_loader_workers,
        persistent_workers=persistent_workers,
    )
    evaluate_fn = _build_evaluate_fn(
        trainer_count=trainer_count,
        batch_size=batch_size,
        dataset_name=dataset_name,
        data_dir=data_dir,
        max_train_samples_per_trainer=max_train_samples_per_trainer,
        max_test_samples=max_test_samples,
        seed=seed,
        device_name=device_name,
        test_loader_workers=test_loader_workers,
        persistent_workers=persistent_workers,
    )

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=trainer_count,
        min_evaluate_clients=0,
        min_available_clients=trainer_count,
        evaluate_fn=evaluate_fn,
        initial_parameters=ndarrays_to_parameters(get_weights(Net())),
    )

    history = start_simulation(
        client_fn=client_fn,
        num_clients=trainer_count,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": client_num_cpus,
            "num_gpus": client_num_gpus,
        },
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
    )

    accuracy_path = run_dir / "accuracy_by_round.csv"
    final_accuracy, best_accuracy, final_loss = _write_round_metrics(
        history=history,
        destination=accuracy_path,
    )
    return ExperimentResult(
        trainer_count=trainer_count,
        final_accuracy=final_accuracy,
        best_accuracy=best_accuracy,
        final_loss=final_loss,
        run_dir=run_dir,
    )


def _write_summary(results: Sequence[ExperimentResult], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "trainer_count",
                "final_accuracy",
                "best_accuracy",
                "final_loss",
                "run_dir",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.trainer_count,
                    _float_or_blank(result.final_accuracy),
                    _float_or_blank(result.best_accuracy),
                    _float_or_blank(result.final_loss),
                    result.run_dir.as_posix(),
                ]
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a federated trainer-count scaling experiment."
    )
    parser.add_argument(
        "--trainer-counts",
        default="10",
        help="Comma-separated trainer counts to run. Default: 10",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="Number of federated rounds per run. Default: 3",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs per trainer. Default: 1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for local training and centralized evaluation. Default: 32",
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
        help="Dataset to use. 'fake' is useful for offline smoke tests.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory used for dataset downloads/cache. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Directory where experiment logs are written. Default: logs",
    )
    parser.add_argument(
        "--max-train-samples-per-trainer",
        type=int,
        default=None,
        help="Optional cap on train samples per trainer to speed up sweeps.",
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
        help="Random seed used for dataset partitioning. Default: 42",
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
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.train_loader_workers < 0 or args.test_loader_workers < 0:
        raise ValueError("DataLoader worker counts must be non-negative.")

    trainer_counts = _parse_trainer_counts(args.trainer_counts)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = Path(args.output_dir) / f"trainer-scaling-{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []
    for trainer_count in trainer_counts:
        print(f"Running experiment with {trainer_count} trainers...")
        result = _run_single_experiment(
            trainer_count=trainer_count,
            num_rounds=args.num_rounds,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            max_train_samples_per_trainer=args.max_train_samples_per_trainer,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
            device_name=args.device,
            client_num_cpus=args.client_num_cpus,
            client_num_gpus=args.client_num_gpus,
            train_loader_workers=args.train_loader_workers,
            test_loader_workers=args.test_loader_workers,
            persistent_workers=args.persistent_workers,
            output_root=output_root,
        )
        results.append(result)
        print(
            "  final_accuracy="
            f"{_float_or_blank(result.final_accuracy) or 'n/a'}"
            " best_accuracy="
            f"{_float_or_blank(result.best_accuracy) or 'n/a'}"
        )
        print(f"  logs: {result.run_dir.as_posix()}")

    summary_path = output_root / "summary.csv"
    _write_summary(results, summary_path)
    print(f"Experiment summary: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
