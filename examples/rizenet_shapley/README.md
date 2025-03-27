---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [tensorflow]
---

# Federated Learning with Rizemind deployed on Rizenet

## Set up

0. Install uv
   **Linux/MacOS**:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**:

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

1. Create a venv

```shell
uv venv .venv_rizenet_shapley
```

2. Activate the python venv environment
   **Linux/MacOS**:

```shell
source .venv_rizenet_shapley/bin/activate
```

**windows**:

```bash
.rizenet_shapley\Scripts\activate
```

3. Install the dependencies defined in `pyproject.toml`.

```bash
uv pip install -e .
```

4. Run the following script and follow the instructions to enable deployment on Rizenet.

```shell
python3 .venv_rizenet_shapley/generate_mnemonic.py
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=5,learning-rate=0.05
```
