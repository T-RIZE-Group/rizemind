"""Decentralized Shapley value strategy for federated learning.

This module provides a decentralized implementation of the Shapley value strategy
where coalition evaluation is distributed to clients rather than performed on the server.
"""

from rizemind.strategies.contribution.shapley.decentralized.shapley_value_client import (
    DecentralShapleyValueClient,
)
from rizemind.strategies.contribution.shapley.decentralized.shapley_value_strategy import (
    DecentralShapleyValueStrategy,
)

__all__ = ["DecentralShapleyValueClient", "DecentralShapleyValueStrategy"]
