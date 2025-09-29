from typing import Protocol

from eth_typing import ChecksumAddress


class SupportsDistribute(Protocol):
    """Protocol for objects that can distribute compensation to federated learning trainers.

    This protocol defines the interface for reward distribution mechanisms in a
    federated learning system. Implementers are responsible for distributing
    compensation to trainers based on their participation scores.

    The protocol supports blockchain-based reward distribution where trainers
    are identified by their Ethereum addresses.
    """

    def distribute(self, trainer_scores: list[tuple[ChecksumAddress, float]]) -> str:
        """Distribute compensation to trainers based on their scores.

        Args:
            trainer_scores: List of tuples containing trainer addresses and their
            corresponding compensation scores. Addresses must be valid Ethereum
            checksum addresses.

        @TODO: Include return value explanation
        """
        ...
