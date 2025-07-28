from abc import ABC, abstractmethod

from eth_typing import ChecksumAddress


class SupportsTrainerContributed(ABC):
    @abstractmethod
    def get_latest_contribution(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> float | None:
        """
        Return the latest contribution value for a trainer, or None if not found.
        """
        pass

    @abstractmethod
    def get_latest_contribution_log(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> dict | None:
        """
        Return the latest contribution log for a trainer, or None if not found.
        """
