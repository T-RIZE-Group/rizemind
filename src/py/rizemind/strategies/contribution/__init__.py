"""Contribution evaluation strategies for federated learning.

This package provides tools for evaluating and quantifying participant
contributions in federated learning scenarios. It enables fair assessment
of how much each participant contributes to the overall model performance,
which is essential for compensation, quality control, and incentive mechanisms.

The package is organized into three main submodules:

- calculators: Contribution calculators, such as Shapley value calculators.
- sampling: Strategies for sampling trainer sets during contribution
  evaluation processes.
- shapley: Comprehensive Shapley value implementation including centralized
  and decentralized strategies.
"""

from rizemind.strategies.contribution import calculators, sampling, shapley

__all__ = ["calculators", "sampling", "shapley"]
