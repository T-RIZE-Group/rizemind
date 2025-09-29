"""Compensation strategies for federated learning participants.

This module provides mechanisms for distributing rewards to participants in
federated learning swarms based on their contributions. It includes protocols
for defining compensation interfaces and concrete implementations for different
reward distribution schemes.

The compensation system operates on blockchain-based addresses and integrates
with Flower federated learning strategies to automatically distribute rewards
after each training round.
"""

from rizemind.strategies.compensation.simple_compensation_strategy import (
    SimpleCompensationStrategy,
)

__all__ = ["SimpleCompensationStrategy"]
