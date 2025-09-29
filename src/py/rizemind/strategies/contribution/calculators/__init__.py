"""Contribution calculator implementations for federated learning.

This package provides algorithms for calculating participant contributions
in federated learning scenarios. Contribution calculators assess how much
each participant contributed to the overall model performance, enabling
fair compensation and quality assessment.
"""

from rizemind.strategies.contribution.calculators.shapley_value import (
    ShapleyValueCalculator,
)

__all__ = ["ShapleyValueCalculator"]
