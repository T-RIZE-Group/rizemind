import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount
from web3.contract import Contract

abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class ERC20(BaseContract, HasAccount):
    """Base ERC20 contract class with standard token functionality."""

    def __init__(self, contract: Contract, *, account: BaseAccount | None = None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    @staticmethod
    def from_address(**kwargs: Unpack[FromAddressKwargs]) -> "ERC20":
        return ERC20(contract_factory(**kwargs, abi=abi))

    def balance_of(self, account: ChecksumAddress) -> int:
        """Get the token balance of an account."""
        return self.contract.functions.balanceOf(account).call()

    def total_supply(self) -> int:
        """Get the total supply of tokens."""
        return self.contract.functions.totalSupply().call()

    def name(self) -> str:
        """Get the token name."""
        return self.contract.functions.name().call()

    def symbol(self) -> str:
        """Get the token symbol."""
        return self.contract.functions.symbol().call()

    def decimals(self) -> int:
        """Get the token decimals."""
        return self.contract.functions.decimals().call()

    def allowance(self, owner: ChecksumAddress, spender: ChecksumAddress) -> int:
        """Get the allowance of spender for owner's tokens."""
        return self.contract.functions.allowance(owner, spender).call()

    def approve(self, spender: ChecksumAddress, amount: int) -> HexBytes:
        """Approve spender to spend amount of tokens."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.approve(spender, amount),
            from_account=account,
        )

    def transfer(self, to: ChecksumAddress, amount: int) -> HexBytes:
        """Transfer amount of tokens to address to."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.transfer(to, amount),
            from_account=account,
        )

    def transfer_from(
        self, from_address: ChecksumAddress, to: ChecksumAddress, amount: int
    ) -> HexBytes:
        """Transfer amount of tokens from from_address to to."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.transferFrom(from_address, to, amount),
            from_account=account,
        )
