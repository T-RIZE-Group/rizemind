"""Containerized training step — the ONLY component that touches raw data.

This is the entrypoint baked into the training container image. It is invoked by
the Prefect flow (as a Docker/Kubernetes job in production) with a handful of URIs
and short-lived, tightly-scoped storage credentials. It:

1. reads the ETL-exported dataset from the (read-only-to-this-step) dataset bucket,
2. reads the global input weights from the weights-in bucket,
3. trains locally,
4. writes the updated weights to the weights-out bucket (readable by the trainer),
5. logs metrics to MLflow.

It never returns weights over the wire and never sees the aggregator or the chain —
it only sees three bucket grants. That isolation is the point of delegation: the
trainer's small orchestration node holds the swarm identity/keys, while the heavy,
data-touching work runs in an ephemeral, least-privilege container on the trainer's
own infrastructure.

Run standalone (what the container does):
    python -m src.training_step \
        --dataset-uri round-1/dataset.npz \
        --weights-in-uri round-1/weights_in.npz \
        --weights-out-uri round-1/weights_out.npz \
        --round 1 --epochs 1 --lr 0.01 \
        --store-backend local --store-location .delegation_bucket \
        --mlflow-uri ./mlruns --mlflow-experiment prefect-delegation
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from .storage import make_store
from .task import (
    Net,
    get_weights,
    load_partition,
    set_weights,
    weights_from_bytes,
    weights_to_bytes,
)


def train_local(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    net.train()
    last_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(net(x), y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    net.eval()
    with torch.no_grad():
        preds = net(x_val).argmax(dim=1)
        accuracy = float((preds == y_val).float().mean().item())
    return {"train_loss": last_loss, "val_accuracy": accuracy}


def run_training_step(args: argparse.Namespace) -> dict[str, float]:
    store = make_store(args.store_backend, args.store_location)

    dataset_bytes = store.get_bytes(args.dataset_uri)
    x_train, y_train, x_val, y_val = load_partition(dataset_bytes)

    net = Net(n_features=x_train.shape[1])
    set_weights(net, weights_from_bytes(store.get_bytes(args.weights_in_uri)))

    start = time.perf_counter()
    metrics = train_local(net, x_train, y_train, x_val, y_val, args.epochs, args.lr)
    metrics["training_time"] = time.perf_counter() - start
    metrics["num_examples"] = float(len(x_train))

    store.put_bytes(args.weights_out_uri, weights_to_bytes(get_weights(net)))

    _log_to_mlflow(args, metrics)
    print(  # container stdout is the flow's log surface
        f"[training-step] round={args.round} "
        f"loss={metrics['train_loss']:.4f} acc={metrics['val_accuracy']:.4f} "
        f"-> {args.weights_out_uri}"
    )
    return metrics


def _log_to_mlflow(args: argparse.Namespace, metrics: dict[str, float]) -> None:
    if not args.mlflow_uri:
        return
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run_name = f"trainer-round-{args.round}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"round": args.round, "epochs": args.epochs, "lr": args.lr})
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=args.round)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delegated containerized training step"
    )
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--weights-in-uri", required=True)
    parser.add_argument("--weights-out-uri", required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--store-backend", default="local")
    parser.add_argument("--store-location", default=".delegation_bucket")
    parser.add_argument("--mlflow-uri", default="")
    parser.add_argument("--mlflow-experiment", default="prefect-delegation")
    return parser


if __name__ == "__main__":
    run_training_step(build_parser().parse_args())
