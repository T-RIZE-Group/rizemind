import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from web3 import Web3
from web3.contract import Contract

from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount

abi = load_abi(Path(os.path.dirname(__file__)) / "./access_control_5_2_0.json")


class AccessControl(BaseContract, HasAccount):
    """OpenZeppelin AccessControl contract wrapper."""

    def __init__(self, contract: Contract, *, account: BaseAccount | None = None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    @staticmethod
    def from_address(
        *, account: BaseAccount | None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "AccessControl":
        return AccessControl(contract_factory(**kwargs, abi=abi), account=account)

    def default_admin_role(self) -> HexBytes:
        """Get the default admin role hash."""
        return self.contract.functions.DEFAULT_ADMIN_ROLE().call()

    def get_role_admin(self, role: HexBytes) -> HexBytes:
        """Get the admin role that controls the specified role."""
        return self.contract.functions.getRoleAdmin(role).call()

    def grant_role(self, role: HexBytes, account: ChecksumAddress) -> HexBytes:
        """Grant a role to an account."""
        sender_account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.grantRole(role, account),
            from_account=sender_account,
        )

    def has_role(self, role: HexBytes, account: ChecksumAddress) -> bool:
        """Check if an account has a specific role."""
        return self.contract.functions.hasRole(role, account).call()

    def renounce_role(
        self, role: HexBytes, caller_confirmation: ChecksumAddress
    ) -> HexBytes:
        """Renounce a role for the calling account."""
        sender_account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.renounceRole(role, caller_confirmation),
            from_account=sender_account,
        )

    def revoke_role(self, role: HexBytes, account: ChecksumAddress) -> HexBytes:
        """Revoke a role from an account."""
        sender_account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.revokeRole(role, account),
            from_account=sender_account,
        )

    def supports_interface(self, interface_id: str) -> bool:
        """Check if the contract supports a specific interface."""
        return self.contract.functions.supportsInterface(interface_id).call()

    def get_role(self, role_name: str) -> HexBytes:
        """Get the keccak256 hash of a role name."""
        return Web3.keccak(text=role_name)
