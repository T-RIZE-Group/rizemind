# Prefect delegation — training on the trainer's own infrastructure

This example shows a **trainer that does not train in-process**. A small
orchestration node holds the swarm identity and talks to the aggregator, but the
actual training is **delegated to a Prefect workflow running on the trainer's own
ETL/compute plane**, inside a least-privilege container that is the only component
allowed to touch raw data. Metrics from both sides go to **MLflow**.

This is the recommended shape for real deployments: the party that owns the data
never ships it anywhere, runs training on infrastructure they control, and hands
back only signed model weights.

## Why delegate?

The default Rizemind examples train inside the Flower client process. That is fine
for simulation, but a real data owner wants to:

- keep raw data on their own infrastructure, behind their own IAM;
- run the heavy training job on their own compute (spot nodes, a K8s cluster, a GPU
  box) that they provision and pay for;
- expose only a **small, long-lived orchestration node** to the federation, holding
  the Ethereum key — not the data, not the GPUs;
- prove the training step they ran was the approved one.

## Architecture

```
                    AGGREGATOR (federation)
                          │  global weights + EIP-712 challenge
                          ▼
   TRAINER INFRASTRUCTURE ─────────────────────────────────────────────
   ┌───────────────────────────┐
   │ Trainer orchestration node│  (Flower client + swarm key; NO data, NO GPU)
   │  src/client.py            │
   │   1. put global weights ──┼──────────────►  [ weights-in bucket ]
   │   2. trigger Prefect flow  │
   │   4. wait, read result ◄──┼──────────────  [ weights-out bucket ]
   │   5. return signed weights │
   └─────────────┬─────────────┘
                 │ run_deployment(...)
                 ▼
   ┌───────────────────────────┐        ETL export (offline)
   │ Prefect flow  src/flow.py │        ┌──────────────────────┐
   │  • provision resources    │        │ Trainer ETL          │
   │  • verify image digest ◄──┼────────┤  writes dataset ─────┼──►[ dataset bucket ]
   │  • run training container │        └──────────────────────┘   (read-only to step)
   └─────────────┬─────────────┘
                 ▼  Docker / K8s job, scoped credentials only
   ┌───────────────────────────────────────────────┐
   │ Containerized training step  src/training_step │
   │  reads: dataset bucket, weights-in bucket      │
   │  writes: weights-out bucket                    │
   │  logs: MLflow                                  │
   │  (no swarm key, no chain, least privilege)     │
   └───────────────────────────────────────────────┘
```

**Access scoping.** The three buckets (or three prefixes) carry different IAM
policies. The ETL can write the dataset; only the training step can read it. The
trainer can write weights-in; only the training step can read it. Only the training
step can write weights-out; the trainer reads it back. The container receives
short-lived credentials (STS token / workload identity) granting exactly those
three scopes — see `src/storage.py`.

**Format compatibility.** The ETL exports `.npz` with `x_train/y_train/x_val/y_val`
(see `make_synthetic_export` in `src/task.py`), which the containerized step loads
directly — no conversion inside the container.

## Additional security step — training-image attestation

> **Note:** the request describing this example was cut off at "an additional
> security step". The mechanism below is our inferred default — please confirm or
> correct it.

The Prefect flow refuses to run any container whose image **digest** is not the
approved one (`verify_image_attestation` in `src/flow.py`). In production the
sanctioned digest is published by the aggregator/DAO — e.g. registered on-chain via
the swarm's `CertificateRegistry` — so a trainer cannot silently substitute a
tampered training step, and the aggregator can audit which image produced each
update. Configure it via `tool.delegation.approved_image_digest` in
`pyproject.toml`. (Left empty in the demo, which runs the step as a subprocess.)

## Components

| File | Role |
|---|---|
| `src/server.py` | Aggregator: deploys the swarm, runs decentralized Shapley, logs to MLflow. |
| `src/client.py` | Trainer orchestration node: delegates `fit` to Prefect, waits, returns signed weights. |
| `src/flow.py` | Prefect flow: provision → attest image → run training container → return artifact URI. |
| `src/training_step.py` | Containerized training entrypoint — the only component that reads raw data. |
| `src/storage.py` | `ArtifactStore` abstraction (local FS by default; swap for S3/GCS). |
| `src/task.py` | Model, weight (de)serialization, ETL-export format. |
| `Dockerfile` | Builds the training-step image (pin its digest as the approved step). |

## Running the demo

Prerequisites: a local blockchain with the Rizemind contracts deployed (see the top
level [`examples/README.md`](../README.md) → *Using Local Blockchain*). Then:

```bash
cd examples/prefect_delegation

# optional: watch MLflow while it runs
uv run -- mlflow ui --backend-store-uri ./mlruns   # http://127.0.0.1:5000

# run the federation; trainers delegate each round to the Prefect flow
uv run -- flwr run .
```

The demo runs the flow in-process and the training step as a subprocess, so no
Docker/Prefect-server is required to see the end-to-end delegation. Artifacts land
under `.delegation_bucket/` (the emulated buckets); metrics land in `./mlruns`.

## Going to production

Replace three seams, nothing else:

1. **Storage** — implement an S3/GCS `ArtifactStore` in `src/storage.py` and set
   `store_backend`/`store_location`; apply the per-scope IAM policies above.
2. **Flow trigger** — swap the in-process `delegated_training(...)` call in
   `src/client.py` for `prefect.deployments.run_deployment(...)` against a Prefect
   **work pool** on the trainer's infra, and register a deployment for the flow.
3. **Training runtime** — swap the subprocess in `src/flow.py` for a Prefect
   `DockerContainer`/`KubernetesJob` block running the image built from `Dockerfile`,
   pinned by digest, with scoped credentials. Register that digest as the approved
   training step.

The trainer's ETL owns the schedule that keeps the dataset bucket fresh; the
federation only ever triggers the flow and reads back signed weights.
