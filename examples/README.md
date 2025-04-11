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

Some examples rely on Rizenet. For setup and usage instructions specific to Rizenet examples, refer to the individual example's README file.

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
