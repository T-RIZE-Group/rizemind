"""The networks this example can run against.

One knob — ``arc-network`` in ``[tool.flwr.app.config]`` — picks the chain, and
everything chain-specific follows from it: the chain ID the run is checked
against, the default RPC endpoint, and the links used to inspect the result.
``local`` is the same code against Anvil, for rehearsing without spending.

Keeping the chain ID and the endpoint in one record is the point. They were two
independent settings before, so a half-applied switch could aim a mainnet chain
guard at a testnet endpoint and fail confusingly, or worse, the reverse.
"""

import os
from dataclasses import dataclass

from rizemind.web3.chains import ARC_MAINNET_CHAINID, ARC_TESTNET_CHAINID

RPC_URL_ENV = "ARC_RPC_URL"
ANVIL_CHAINID = 31337


@dataclass(frozen=True)
class ArcNetwork:
    """An Arc network and the facts that depend on which one it is.

    Attributes:
        name: The value `arc-network` is set to, such as ``"mainnet"``.
        chain_id: The chain ID the run is checked against before it spends.
        default_rpc_url: The endpoint used unless `$ARC_RPC_URL` overrides it.
        explorer_url: Base URL of the chain's explorer, empty when it has none,
            as Anvil does not.
        gas_source: Where the gas comes from — worth printing, since one of
            these networks costs real money.
    """

    name: str
    chain_id: int
    default_rpc_url: str
    explorer_url: str
    gas_source: str

    @property
    def is_mainnet(self) -> bool:
        """Whether this network is Arc Mainnet."""
        return self.chain_id == ARC_MAINNET_CHAINID

    @property
    def rpc_url(self) -> str:
        """The endpoint to use, letting `$ARC_RPC_URL` override the default.

        A private or rate-limited endpoint is a deployment detail, not a
        property of the network, so it belongs in the environment rather than
        in `pyproject.toml`.

        Returns:
            `$ARC_RPC_URL` when set and non-empty, otherwise this network's
            default endpoint.
        """
        return os.environ.get(RPC_URL_ENV) or self.default_rpc_url

    def explorer_address_url(self, address: str) -> str:
        """Build an explorer link for a contract or account.

        Args:
            address: The address to link to.

        Returns:
            The explorer URL, or an empty string when the network has no
            explorer. Callers should skip the link rather than print an empty
            one.
        """
        if not self.explorer_url:
            return ""
        return f"{self.explorer_url}/address/{address}"


MAINNET = ArcNetwork(
    name="mainnet",
    chain_id=ARC_MAINNET_CHAINID,
    default_rpc_url="https://rpc.mainnet.arc.io",
    explorer_url="https://explorer.arc.io",
    gas_source="real USDC",
)

TESTNET = ArcNetwork(
    name="testnet",
    chain_id=ARC_TESTNET_CHAINID,
    default_rpc_url="https://rpc.testnet.arc.io",
    explorer_url="https://explorer.testnet.arc.io",
    gas_source="test USDC from https://faucet.circle.com",
)

LOCAL = ArcNetwork(
    name="local",
    chain_id=ANVIL_CHAINID,
    default_rpc_url="http://127.0.0.1:8545",
    explorer_url="",
    gas_source="free (Anvil)",
)

NETWORKS = {network.name: network for network in (MAINNET, TESTNET, LOCAL)}


def get_network(name: str) -> ArcNetwork:
    """Look up a network by name.

    Args:
        name: ``"mainnet"``, ``"testnet"`` or ``"local"``. Case and surrounding
            whitespace are ignored.

    Returns:
        The matching network record.

    Raises:
        ValueError: If the name is not a known network. Failing here beats
            defaulting to one of them, since they differ by real money.
    """
    try:
        return NETWORKS[name.strip().lower()]
    except KeyError:
        known = ", ".join(sorted(NETWORKS))
        raise ValueError(
            f"unknown network {name!r}; expected one of: {known}"
        ) from None
