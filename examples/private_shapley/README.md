# Private Shapley Value Example

This example demonstrates a private implementation of Shapley value calculation for federated learning using bitfields and Merkle proofs.

## Overview

This implementation uses the `PrivateShapley` smart contract to provide:

1. **Trainer Registration**: Each trainer is registered with a unique index
2. **Coalition Formation**: Coalitions are represented using bitfields where each bit corresponds to a trainer
3. **Commitment Scheme**: Trainers use a commitment scheme with nonces for privacy
4. **Merkle Proofs**: Merkle trees are used to verify trainer participation without revealing the entire coalition
5. **Reward Distribution**: Trainers can claim rewards by providing valid Merkle proofs

## Requirements

- Local Ethereum node (Anvil)
- Deployed PrivateShapley contract
- Python 3.12+

## Setup Instructions

1. Install dependencies:
   ```bash
   uv sync
   ```
