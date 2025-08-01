from typing import Protocol

from eth_typing import ChecksumAddress

from rizemind.contracts.erc.erc5267.typings import EIP712Domain


class SupportsEthAccountStrategy(Protocol):
    def can_train(self, trainer: ChecksumAddress, round_id: int) -> bool: ...
    def get_eip712_domain(self) -> EIP712Domain: ...
