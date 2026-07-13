"""Model, weight (de)serialization, and dataset loading for the delegated example.

Kept deliberately small and dependency-light (a compact MLP over a synthetic
tabular dataset) so the focus stays on the *delegation architecture* rather than
the model. The dataset stands in for whatever the trainer's ETL exports.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from torch import nn


class Net(nn.Module):
    """A small MLP classifier — placeholder for the trainer's real model."""

    def __init__(self, n_features: int = 20, n_classes: int = 2) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def get_weights(net: nn.Module) -> list[np.ndarray]:
    return [val.cpu().numpy() for val in net.state_dict().values()]


def set_weights(net: nn.Module, weights: list[np.ndarray]) -> None:
    params = zip(net.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params}
    net.load_state_dict(state_dict, strict=True)


def weights_to_bytes(weights: list[np.ndarray]) -> bytes:
    """Serialize a parameter list to a portable ``.npz`` blob for the buckets."""
    buffer = io.BytesIO()
    np.savez(buffer, *weights)
    return buffer.getvalue()


def weights_from_bytes(data: bytes) -> list[np.ndarray]:
    buffer = io.BytesIO(data)
    with np.load(buffer) as npz:
        return [npz[key] for key in npz.files]


def load_partition(
    dataset_bytes: bytes,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load an ETL-exported dataset blob into train/val tensors.

    The blob is a ``.npz`` with ``x_train/y_train/x_val/y_val`` — the format the
    ETL export step is expected to produce so it is directly compatible with the
    containerized training step (no conversion inside the container).
    """
    buffer = io.BytesIO(dataset_bytes)
    with np.load(buffer) as npz:
        x_train = torch.tensor(npz["x_train"], dtype=torch.float32)
        y_train = torch.tensor(npz["y_train"], dtype=torch.long)
        x_val = torch.tensor(npz["x_val"], dtype=torch.float32)
        y_val = torch.tensor(npz["y_val"], dtype=torch.long)
    return x_train, y_train, x_val, y_val


def make_synthetic_export(
    seed: int, n_features: int = 20, n_train: int = 512, n_val: int = 128
) -> bytes:
    """Stand-in for the trainer's ETL: produce a per-trainer dataset export.

    Each trainer gets a differently-seeded, slightly-shifted distribution so that
    contribution scoring (Shapley) has something real to measure.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n_features,))
    shift = rng.normal(scale=0.3, size=(n_features,))

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, n_features)) + shift
        logits = x @ w
        y = (logits > np.median(logits)).astype(np.int64)
        return x.astype(np.float32), y

    x_train, y_train = sample(n_train)
    x_val, y_val = sample(n_val)
    buffer = io.BytesIO()
    np.savez(buffer, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    return buffer.getvalue()
