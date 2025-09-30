"""Federated learning strategies for contribution evaluation and compensation.

This package provides the core strategies for Rizemind's federated
learning framework. It includes mechanisms for both evaluating participant
contributions and distributing rewards based on those contributions.

The package contains two main submodules:

- compensation: Strategies for distributing rewards to participants based on
  their contributions. These integrate with blockchain-based payment systems
  and Flower federated learning strategies.
- contribution: Tools for evaluating and quantifying how much each participant
  contributes to the overall model performance. This includes Shapley value
  calculations, sampling strategies, and both centralized and decentralized
  evaluation approaches.

Together, these strategies enable fair and transparent federated learning by
connecting contribution measurement to appropriate compensation mechanisms.
"""

from rizemind.strategies import compensation, contribution

__all__ = ["compensation", "contribution"]
