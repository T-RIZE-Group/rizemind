from typing import Protocol

from eth_typing import ChecksumAddress


class SupportsDistribute(Protocol):
    def distribute(
        self, round_id: int, trainer_scores: list[tuple[ChecksumAddress, float]]
    ) -> str: ...
