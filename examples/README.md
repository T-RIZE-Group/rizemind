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

## Using Local Blockchain

To run a local blockchain, we are going to use **Foundry**. [Foundry](https://book.getfoundry.sh/) is a comprehensive smart contract development toolchain that manages dependencies, compiles projects, runs tests, deploys contracts, and facilitates interactions with the blockchain via command-line or Solidity scripts.

Several examples utilize Foundry to create a local testing environment for contracts before deployment. To set up Foundry in this repository:

1. Navigate to the [forge directory](https://github.com/T-RIZE-Group/rizemind/tree/main/forge).
2. Follow the instructions provided in the README within that directory.

Ensure your local blockchain is running before executing examples requiring a local blockchain.

---

## Using Rizenet

Some examples rely on **Rizenet testnet**. Rizenet provides **Factories**, which enable an aggregator to easily instantiate new **smart contracts** for **access control** and **compensation distribution** in decentralized machine learning systems.

For setup and usage instructions specific to Rizenet examples, follow the steps below:

### Set up

#### Setup the repository

First make sure you have properly followed the repository by following the main "for developers" section in our documents.

#### Generate mnemonics

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

#### Save Your Mnemonic
   To securely store your mnemonic, copy `env.example` to `.env`:

```bash
cp env.example .env
```

Then, **edit the `.env` file** and replace the placeholder with your generated mnemonic:

```text
RIZENET_MNEMONIC="picture fine relief success curious avocado define divert cause genuine such master"
```

> ![CAUTION] **Never share your mnemonic with anyone.** It grants full access to your accounts. Store it securely and do not commit `.env` files to version control.

#### Whitelist Aggregator

Rizenet is a **permissioned network** for **smart contract deployment**.
Before deploying contracts, you must **whitelist your aggregator address**.

- Go to `Rizenet Deployer <https://rizenet.io/deployer>`\_.
- Enter your **aggregator address** (generated earlier).
- Click "Enable" and wait for confirmation.

Once approved, your aggregator will have permission to deploy smart contracts.

#### Get Gas

Blockchain transactions require **gas** to process computations.
On the **Rizenet Testnet**, gas is **free**, but you need to request testnet tokens. To get gas:

- Visit `Rizenet Faucet <https://rizenet.io/faucets>`\_.
- Enter your **aggregator address**.
- Click drip and wait for confirmation.

Once your aggregator has gas, you're ready to deploy smart contracts.

#### Run the project

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
