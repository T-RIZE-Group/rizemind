"""Centralized Shapley value strategy for federated learning.

This module provides a centralized implementation of the Shapley value strategy
where model evaluation is performed on the server side rather than by clients.
"""

from rizemind.strategies.contribution.shapley.centralized.shapley_value_strategy import (
    CentralShapleyValueStrategy,
)

__all__ = ["CentralShapleyValueStrategy"]
