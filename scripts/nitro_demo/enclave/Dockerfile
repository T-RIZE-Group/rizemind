FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir cryptography

COPY scripts/nitro_demo/common.py /app/scripts/nitro_demo/common.py
COPY scripts/nitro_demo/framing.py /app/scripts/nitro_demo/framing.py
COPY scripts/nitro_demo/03_enclave_server.py /app/scripts/nitro_demo/03_enclave_server.py
COPY data/nitro_demo/keys.json /app/data/nitro_demo/keys.json

ENTRYPOINT ["python", "/app/scripts/nitro_demo/03_enclave_server.py"]
