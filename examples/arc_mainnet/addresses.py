"""Print the addresses this example derives, without touching the network.

    export RZMND_PASSPHRASE='…'
    uv run -- python addresses.py

Account 0 of the configured mnemonic is the aggregator — it is the only account
that sends transactions, so it is the only one that needs USDC. Accounts 1..N
are the trainers; they sign EIP-712 messages off-chain and stay at zero balance.

Nothing here prints a mnemonic, a private key, or the keystore passphrase.
"""

import sys

from eth_account import Account
from rizemind.authentication.config import AccountConfig
from rizemind.configuration.toml_config import TomlConfig


def main() -> int:
    Account.enable_unaudited_hdwallet_features()

    config = TomlConfig("./pyproject.toml")
    num_supernodes = int(config.get("tool.flwr.app.config.num-supernodes"))

    try:
        account_config = AccountConfig(**config.get("tool.eth.account"))
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
        print(f"cannot load the account: {exc}", file=sys.stderr)
        print(
            "\nGenerate one first:\n"
            "  uv run -- rzmnd account generate arc-aggregator --words 24\n"
            "  export RZMND_PASSPHRASE='the passphrase you chose'",
            file=sys.stderr,
        )
        return 1

    print(f"aggregator (fund this one): {account_config.get_account(0).address}")
    for i in range(1, num_supernodes + 1):
        print(
            f"trainer {i} (no funding needed): {account_config.get_account(i).address}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
