# Examples

This directory contains practical examples demonstrating how to effectively use the Rizemind library. Each example includes clear, step-by-step instructions to help you quickly set up and run distributed machine learning tasks.

## How to Run an Example

To run the examples, generally you need to navigate to its directory, setup a few steps, and execute the simulation server using:

```bash
cd <example_dir>
uv run -- flwr run .
```

> **Note:** Some examples may require additional prerequisites. Always check the specific README file inside each example directory for detailed instructions.

There are two types of examples, some of them use a local blockchain, and some use the rizenet testnet. To learn how to setup each example properly, read below.

---

### Using Local Blockchain

To run a local blockchain, we are going to use **Foundry**. [Foundry](https://book.getfoundry.sh/) is a comprehensive smart contract development toolchain that manages dependencies, compiles projects, runs tests, deploys contracts, and facilitates interactions with the blockchain via command-line or Solidity scripts.

Several examples utilize Foundry to create a local testing environment for contracts before deployment. To set up Foundry in this repository, follow the steps below:

0. Install [foundry](https://book.getfoundry.sh/getting-started/installation)
1. Navigate to the forge directory.
2. Run the following commands line by line:

```bash
# Install foundry tools
foundryup

# Install soldeer
forge soldeer install

# Build the project's smart contracts
forge build

# Run local ethereum node
anvil

# Publish the contracts to the local blockchain
forge script script/deployments/SwarmV1Factory.s.sol --rpc-url http://127.0.0.1:8545 --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 --broadcast
```

Ensure your local blockchain is running before executing examples requiring a local blockchain.

> ![CAUTION] in the `forge script` command, the provided private key above is disposable. In real world, never commit or publish your real private key into your repository.

---

### Using Rizenet

Some examples rely on **Rizenet testnet**. Rizenet provides **Factories**, which enable an aggregator to easily instantiate new **smart contracts** for **access control** and **compensation distribution** in decentralized machine learning systems.

For setup and usage instructions specific to Rizenet examples, follow the steps below:

#### Set up

##### Setup the repository

First make sure you have properly followed the repository by following the main "for developers" section in our documents.

##### Generate mnemonics

Generate 12-word mnemonics by running:

```bash
uv run -- python {project_name}/generate_mnemonic.py
```

the output will look something like this:

```text
Generated Mnemonic Phrase:
    picture fine relief success curious avocado define divert cause genuine such master

    Copy the `env.example` to `.env` and replace the RIZENET_MNEMONIC with the mnemonic above.

    Your aggregator address: 0x16E5D170372bDEF845dE2192429e822900F7592F

    To enable your aggregator to launch contracts on Rizenet:
    1. Enable smart contract deployment: https://rizenet.io/deployer
    2. Get gas: https://rizenet.io/faucets
```

The two most important pieces of information here are:

- **The 12-word mnemonic phrase** (used to derive private keys).
- **The aggregator address** (used to deploy contracts).

##### Save Your Mnemonic

To securely store your mnemonic, copy `env.example` to `.env`:

```bash
cp env.example .env
```

Then, **edit the `.env` file** and replace the placeholder with your generated mnemonic:

```text
RIZENET_MNEMONIC="picture fine relief success curious avocado define divert cause genuine such master"
```

> ![CAUTION] **Never share your mnemonic with anyone.** It grants full access to your accounts. Store it securely and do not commit `.env` files to version control.

##### Whitelist Aggregator

Rizenet is a **permissioned network** for **smart contract deployment**.
Before deploying contracts, you must **whitelist your aggregator address**.

- Go to `Rizenet Deployer <https://rizenet.io/deployer>`\_.
- Enter your **aggregator address** (generated earlier).
- Click "Enable" and wait for confirmation.

Once approved, your aggregator will have permission to deploy smart contracts.

##### Get Gas

Blockchain transactions require **gas** to process computations.
On the **Rizenet Testnet**, gas is **free**, but you need to request testnet tokens. To get gas:

- Visit `Rizenet Faucet <https://rizenet.io/faucets>`\_.
- Enter your **aggregator address**.
- Click drip and wait for confirmation.

Once your aggregator has gas, you're ready to deploy smart contracts.

##### Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

- Run with the Simulation Engine

```bash
uv run -- flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
uv run -- flwr run . --run-config num-server-rounds=5,learning-rate=0.05
```

> ![NOTE]: Make sure to always use the command `uv run --` before calling the actual command. This way `uv` will make sure you have the proper dependencies installed.

## How to create a new example

To create a new example, make a new sub-directory in the `examples`, open it in your terminal, and run the following command:

```bash
uv init --python 3.12 # Since our dependencies are currently only compatible with python 3.12
```

This will tell `uv` to add the new example as a workspace, allowing the dependencies to be shared. Please keep in mind that all workspaces share the same dependency version. This is to facilitate dependency management across the examples and the main library.

Now you will see three files under your newly created sub-directory:

1. A `pyproject.toml` file
2. A main.py file
3. A README.md file

Go ahead and remove the main.py file. Then copy the following lines into your `pyproject.toml` file:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["."]
[tool.hatch.metadata]
allow-direct-references = true
```

You also need to add some additional configurations for your flower client/server based on the type of app you are developing. For that please check [flower documentation](https://flower.ai/docs/framework/) or check one of our examples for inspiration.

To add dependencies to your examples, run:

```bash
uv add <package-list>
```

This will install all the packages for your examples. Now if you have configured everything properly, you can simply call `uv run -- flwr run .` for flower to start simulating your example.

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
