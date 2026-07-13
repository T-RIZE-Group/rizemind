"""Artifact storage abstraction for delegated training.

The delegation pattern moves data through *buckets* whose access is scoped per
step, so that the training container can read the exported dataset and the input
weights but nothing else, and the trainer node can read the output weights but not
the raw dataset.

`ArtifactStore` is the seam a trainer replaces with their real object store
(S3, GCS, Azure Blob, MinIO). The default `LocalArtifactStore` writes to a local
directory so the example runs end-to-end without any cloud infrastructure — the
directory layout mirrors the bucket/prefix layout you would use in production.

Access-scoping note
-------------------
In production the three URIs below live in (or under prefixes of) buckets with
*different* IAM policies:

* ``dataset`` — written by the ETL, readable only by the training step.
* ``weights_in`` — written by the trainer, readable only by the training step.
* ``weights_out`` — written by the training step, readable only by the trainer.

The training container is handed short-lived credentials (e.g. an STS token or a
workload-identity binding) that grant exactly those three grants and nothing else.
`LocalArtifactStore` cannot enforce IAM, so it only emulates the *layout*; the
enforcement is a deployment concern documented in the README.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    """Minimal object-store interface used by the trainer and the training step."""

    @abstractmethod
    def put_bytes(self, uri: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, uri: str) -> bytes: ...

    @abstractmethod
    def exists(self, uri: str) -> bool: ...


class LocalArtifactStore(ArtifactStore):
    """Filesystem-backed store. ``uri`` is treated as a path under ``root``.

    Swap this for an S3/GCS implementation in production; the ``uri`` strings
    (``round-3/weights_in.npz`` etc.) become object keys under a bucket/prefix.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, uri: str) -> Path:
        path = self.root / uri
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_bytes(self, uri: str, data: bytes) -> None:
        self._path(uri).write_bytes(data)

    def get_bytes(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def exists(self, uri: str) -> bool:
        return (self.root / uri).exists()

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)


def make_store(backend: str, location: str) -> ArtifactStore:
    """Factory so the backend can be selected from config.

    Extend with ``s3``/``gcs`` branches that return the corresponding
    implementation. The rest of the example only depends on `ArtifactStore`.
    """
    if backend == "local":
        return LocalArtifactStore(location)
    raise ValueError(
        f"Unknown storage backend '{backend}'. "
        "Add an S3/GCS ArtifactStore implementation for production use."
    )
