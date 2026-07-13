"""Prefect flow that runs on the trainer's own infrastructure.

The trainer's small orchestration node does NOT train. It triggers this flow (via
``run_deployment``) on the trainer's ETL/compute plane. The flow provisions
resources, runs the containerized training step against scoped buckets, verifies
the container image, and returns the URI of the produced weights.

In production the ``run_containerized_training`` task would submit a Docker or
Kubernetes job (Prefect's ``DockerContainer`` / ``KubernetesJob`` infrastructure
blocks) pinned to an image *digest*, with a service account granting only the three
bucket scopes. Here it launches the same entrypoint in a subprocess so the example
runs without a container runtime, while preserving the process/credential boundary.
"""

from __future__ import annotations

import subprocess
import sys

from prefect import flow, get_run_logger, task

from .storage import make_store


@task(retries=2, retry_delay_seconds=5)
def provision_resources(round_id: int) -> None:
    """Provision the ephemeral compute for this round.

    A no-op locally. In production this is where a work pool / autoscaling node /
    spot instance is requested before the container runs.
    """
    get_run_logger().info("Provisioning resources for round %s", round_id)


@task
def verify_image_attestation(image_digest: str, approved_digest: str | None) -> str:
    """Additional security step (inferred): only run an *approved* training image.

    The aggregator/DAO registers the digest of the sanctioned training container
    (e.g. on-chain via the swarm CertificateRegistry). The flow refuses to run any
    other image, so a trainer cannot silently substitute a tampered training step.
    This is a placeholder for that check — see the README's "additional security
    step" section, and confirm the intended mechanism.
    """
    logger = get_run_logger()
    if approved_digest and image_digest != approved_digest:
        raise ValueError(
            f"Refusing to run unapproved training image {image_digest!r}; "
            f"expected {approved_digest!r}"
        )
    logger.info("Training image attested: %s", image_digest)
    return image_digest


@task(retries=1)
def run_containerized_training(
    *,
    dataset_uri: str,
    weights_in_uri: str,
    weights_out_uri: str,
    round_id: int,
    epochs: int,
    lr: float,
    store_backend: str,
    store_location: str,
    mlflow_uri: str,
    mlflow_experiment: str,
) -> str:
    """Run the training container. Locally: subprocess; in prod: Docker/K8s job."""
    logger = get_run_logger()
    cmd = [
        sys.executable,
        "-m",
        "src.training_step",
        "--dataset-uri",
        dataset_uri,
        "--weights-in-uri",
        weights_in_uri,
        "--weights-out-uri",
        weights_out_uri,
        "--round",
        str(round_id),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--store-backend",
        store_backend,
        "--store-location",
        store_location,
        "--mlflow-uri",
        mlflow_uri,
        "--mlflow-experiment",
        mlflow_experiment,
    ]
    logger.info("Launching training step for round %s", round_id)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Training step failed:\n%s", result.stderr)
        raise RuntimeError(f"Training step exited {result.returncode}")
    logger.info(result.stdout.strip())

    store = make_store(store_backend, store_location)
    if not store.exists(weights_out_uri):
        raise RuntimeError(f"Training step produced no artifact at {weights_out_uri}")
    return weights_out_uri


@flow(name="delegated-training")
def delegated_training(
    *,
    dataset_uri: str,
    weights_in_uri: str,
    weights_out_uri: str,
    round_id: int,
    epochs: int = 1,
    lr: float = 0.01,
    image_digest: str = "local:subprocess",
    approved_digest: str | None = None,
    store_backend: str = "local",
    store_location: str = ".delegation_bucket",
    mlflow_uri: str = "",
    mlflow_experiment: str = "prefect-delegation",
) -> str:
    """Provision → attest → train → return output artifact URI."""
    provision_resources(round_id)
    verify_image_attestation(image_digest, approved_digest)
    return run_containerized_training(
        dataset_uri=dataset_uri,
        weights_in_uri=weights_in_uri,
        weights_out_uri=weights_out_uri,
        round_id=round_id,
        epochs=epochs,
        lr=lr,
        store_backend=store_backend,
        store_location=store_location,
        mlflow_uri=mlflow_uri,
        mlflow_experiment=mlflow_experiment,
    )
