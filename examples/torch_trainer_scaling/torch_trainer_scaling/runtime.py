from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    normalized = device.strip().lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends.mps, "is_available", lambda: False)():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if normalized == "mps":
        if not getattr(torch.backends.mps, "is_available", lambda: False)():
            raise ValueError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if normalized == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device '{device}'. Available values: auto, cpu, mps, cuda."
    )
