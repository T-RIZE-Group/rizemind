# Containerized training step — the only component that reads raw data.
#
# In production the Prefect flow runs THIS image (pinned by digest) as a Docker or
# Kubernetes job, handed short-lived credentials scoped to exactly three bucket
# grants: read dataset, read weights-in, write weights-out. It has no swarm keys,
# no chain access, and no network egress beyond the object store + MLflow.
#
# Build:  docker build -t <registry>/rizemind-training-step:<tag> .
# The resulting image DIGEST is what you register as the approved training step
# (tool.delegation.approved_image_digest / on-chain attestation).

FROM python:3.12-slim

WORKDIR /app

# Only the pieces the training step needs — deliberately not the full trainer.
RUN pip install --no-cache-dir "torch>=2.7.0" "numpy>=1.26.4" "mlflow>=3.3.2"

COPY src/__init__.py src/__init__.py
COPY src/storage.py src/storage.py
COPY src/task.py src/task.py
COPY src/training_step.py src/training_step.py

# The flow passes bucket URIs and scoped credentials as args/env at run time.
ENTRYPOINT ["python", "-m", "src.training_step"]
