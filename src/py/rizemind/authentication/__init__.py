"""Authentication utilities for federated learning with Ethereum-based accounts.

This package exposes server and client-side helpers used to authenticate
Flower clients and aggregator with EIP-712 signatures and to authorize training based on
on-chain swarm policies.

Typical usage example:

    from rizemind.authentication import EthAccountStrategy, AccountConfig, authentication_mod

    # Server: wrap a base strategy
    eth_strategy = EthAccountStrategy(strat=base_strategy, swarm=swarm, account=AccountConfig(...).get_account(0))

    # Client: add authentication middleware to the app chain
    app = ClientApp(client_fn, mods=[authentication_mod, model_notary_mod])
"""

from rizemind.authentication.authenticated_client_properties import (
    AuthenticatedClientProperties,
)
from rizemind.authentication.client_manager import (
    ClientManagerWithCriterion,
)
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_strategy import EthAccountStrategy
from rizemind.authentication.mod import authentication_mod

from . import notary, signatures

__all__ = [
    "notary",
    "signatures",
    "ClientManagerWithCriterion",
    "AuthenticatedClientProperties",
    "authentication_mod",
    "AccountConfig",
    "EthAccountStrategy",
]
