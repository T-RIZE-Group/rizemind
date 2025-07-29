from getpass import getpass

import typer

from rizemind.authentication.config import AccountConfig, MnemonicStoreConfig


def account_config_loader(mnemonic: str | None = None, account_name: str | None = None):
    if mnemonic is not None:
        return AccountConfig(mnemonic=mnemonic)

    if account_name is not None:
        passphrase = getpass("Account passphrase:")
        return AccountConfig(
            mnemonic_store=MnemonicStoreConfig(
                account_name=account_name, passphrase=passphrase
            )
        )

    typer.echo("⚠️  --mnemonic or --account_name required", err=True)
    raise typer.Exit(code=1)
