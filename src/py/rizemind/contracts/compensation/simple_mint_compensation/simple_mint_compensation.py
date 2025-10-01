import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from web3.contract import Contract

from rizemind.configuration.validators.eth_address import EthereumAddress
from rizemind.contracts.abi import encode_with_selector
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.compensation.compensation_factory import CompensationConfig
from rizemind.contracts.erc.erc20.erc20 import ERC20
from rizemind.contracts.erc.erc5267.erc5267 import ERC5267
from rizemind.contracts.open_zeppelin.access_control import AccessControl


class SimpleMintCompensationConfig(CompensationConfig):
    name: str = "simple-mint-compensation"
    version: str = "1.0.0"

    token_name: str
    token_symbol: str
    target_rewards: int
    initial_admin: EthereumAddress | None = None
    minter: EthereumAddress | None = None

    def get_init_data(self, *, swarm_address: ChecksumAddress) -> HexBytes:
        """
        Generate initialization data for SimpleMintCompensation.

        Args:
            swarm_address: The swarm address (used as default for initial_admin and minter if not specified)

        Returns:
            Encoded initialization data
        """
        return encode_with_selector(
            "initialize(string,string,uint256,address,address)",
            ["string", "string", "uint256", "address", "address"],
            [
                self.token_name,
                self.token_symbol,
                self.target_rewards,
                self.initial_admin or swarm_address,
                self.minter or swarm_address,
            ],
        )


abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class SimpleMintCompensation(ERC20, AccessControl, ERC5267):
    def __init__(self, contract: Contract, *, account: BaseAccount | None = None):
        ERC20.__init__(self, contract=contract, account=account)
        AccessControl.__init__(self, contract=contract, account=account)
        ERC5267.__init__(self, contract=contract)

    @staticmethod
    def from_address(
        *, account: BaseAccount | None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "SimpleMintCompensation":
        return SimpleMintCompensation(
            contract_factory(**kwargs, abi=abi), account=account
        )

    def initialize(
        self,
        token_name: str,
        token_symbol: str,
        target_rewards: int,
        initial_admin: ChecksumAddress,
        minter: ChecksumAddress,
    ) -> HexBytes:
        """Initialize the SimpleMintCompensation contract."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.initialize(
                token_name, token_symbol, target_rewards, initial_admin, minter
            ),
            from_account=account,
        )

    def distribute(
        self,
        round_id: int,
        recipients: list[ChecksumAddress],
        contributions: list[int],
    ) -> HexBytes:
        """Distribute compensation to recipients based on their contributions."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.distribute(
                round_id, recipients, contributions
            ),
            from_account=account,
        )

    def supports_interface(self, interface_id: str) -> bool:
        """Check if the contract supports a specific interface."""
        return self.contract.functions.supportsInterface(interface_id).call()
