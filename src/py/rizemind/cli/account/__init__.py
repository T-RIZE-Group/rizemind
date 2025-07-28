from getpass import getpass
from typing import Annotated

import typer
from eth_account import Account

from rizemind.cli.account.loader import account_config_loader
from rizemind.mnemonic.store import MnemonicStore

Account.enable_unaudited_hdwallet_features()

account = typer.Typer(help="Account management commands")

mnemonic_store = MnemonicStore()


@account.command("generate")
def generate(
    account_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Friendly name for the new account",
        ),
    ],
    words: Annotated[
        int,
        typer.Option(
            "--words",
            "-w",
            help="Mnemonic length; choose 12 or 24",
            rich_help_panel="Security",
        ),
    ] = 12,
) -> None:
    """Generate a mnemonic, encrypt it, and save a keystore file."""

    if words not in (12, 24):
        typer.echo("--words must be 12 or 24.", err=True)
        raise typer.Exit(code=1)

    pwd1 = getpass("Passphrase: ")
    pwd2 = getpass("Confirm passphrase: ")
    if pwd1 != pwd2:
        typer.echo("Passphrases do not match.", err=True)
        raise typer.Exit(1)

    account = mnemonic_store.generate(words=words)
    if mnemonic_store.exists(account_name):
        typer.echo("⚠️  Name already used", err=True)
        raise typer.Exit(code=1)

    file_path = mnemonic_store.save(account_name, pwd1, account)

    typer.echo(f"Saved encrypted mnemonic: {file_path}")


@account.command("list")
def list_accounts() -> None:
    """Show all stored account names."""
    names = mnemonic_store.list_accounts()

    if not names:
        typer.echo("No accounts found.")
        return

    typer.echo("Stored accounts:")
    for name in names:
        typer.echo(f"- {name}")


@account.command("load")
def load_account(
    account_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Name of the stored account to unlock",
        ),
    ],
) -> None:
    """
    Display the stored mnemonic for *ACCOUNT_NAME* and show the first 10
    derived Ethereum addresses.
    """
    account_config = account_config_loader(account_name=account_name)

    typer.echo(f'Mnemonic:\n"{account_config.mnemonic}"\n')

    # Derive and display the first 10 HD-wallet accounts
    typer.echo("First 10 derived addresses:")
    for i in range(10):
        acct = account_config.get_account(i)
        typer.echo(f"  {i:>2}: {acct.address}, private key: {acct.key.hex()}")


if __name__ == "__main__":
    account()
