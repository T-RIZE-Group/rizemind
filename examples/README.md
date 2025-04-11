# Examples

This directory contains practical examples demonstrating how to effectively use the Rizemind library. Each example includes clear, step-by-step instructions to help you quickly set up and run distributed machine learning tasks.

## How to Run an Example

To run any example, navigate to its directory and execute the simulation server using:

```bash
cd <example_dir>
uv run -- flwr run .
```

> **Note:** Some examples may require additional prerequisites. Always check the specific README file inside each example directory for detailed instructions.

---

## Setting up Foundry

[Foundry](https://book.getfoundry.sh/) is a comprehensive smart contract development toolchain that manages dependencies, compiles projects, runs tests, deploys contracts, and facilitates interactions with the blockchain via command-line or Solidity scripts.

Several examples utilize Foundry to create a local testing environment for contracts before deployment. To set up Foundry in this repository:

1. Navigate to the [forge directory](https://github.com/T-RIZE-Group/rizemind/tree/main/forge).
2. Follow the instructions provided in the README within that directory.

Ensure your local blockchain is running before executing examples requiring a local blockchain.

---

## Using Rizenet

Some examples rely on Rizenet. For setup and usage instructions specific to Rizenet examples, follow the steps below:

## Set up

0. Setup the repository

First make sure you have properly followed the repository by following the main "for developers" section in our documents.

1. Generate mnemonics

Generate 12-word mnemonics by running:

```bash
uv run -- python {project_name}/generate_mnemonic.py
```

and copy the mnemonic key into a `.env` file as the following:

```bash
RIZENET_MNEMONIC="your mnemonic"
```

2. Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

- Run with the Simulation Engine

```bash
uv run -- flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
uv run -- flwr run . --run-config num-server-rounds=5,learning-rate=0.05
```

> Note: Make sure to always use the command `uv run --` before calling the actual command. This way `uv` will make sure you have the proper dependencies installed.

---

## Examples Compatibility Overview

The table below clarifies which examples require a local blockchain and which ones are designed for Rizenet:

| Example                     | Local Blockchain | Rizenet |
| --------------------------- | ---------------- | ------- |
| Basic Signature             | ✅               | ❌      |
| Centralized Shapley Value   | ✅               | ❌      |
| Decentralized Shapley Value | ✅               | ❌      |
| Decentralized TabPFN        | ✅               | ❌      |
| RizeNet Deployment          | ❌               | ✅      |
| RizeNet Shapley             | ❌               | ✅      |
