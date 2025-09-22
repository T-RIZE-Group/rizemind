"""
Phase polling module for swarm lifecycle management.

This module provides utilities to monitor and react to phase changes in the swarm,
similar to how the web3 indexer monitors block changes.
"""

from .latest_phase_bus import LatestPhaseBus
from .phase_watcher import PhaseWatcher

__all__ = ["LatestPhaseBus", "PhaseWatcher"]
