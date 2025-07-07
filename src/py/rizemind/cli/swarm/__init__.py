from typing import Annotated, List

import typer
from pydantic import HttpUrl

from rizemind.cli.account.loader import account_config_loader
from rizemind.cli.account.options import AccountNameOption, MnemonicOption
from rizemind.contracts.models.model_factory_v1 import (
    ModelFactoryV1,
    ModelFactoryV1Config,
)
from rizemind.web3.config import Web3Config

swarm = typer.Typer(help="Federation management commands")


def csv_to_list(
    _ctx: typer.Context,
    _param: typer.CallbackParam,
    value: List[str],
) -> List[str]:
    """Split a comma-separated string into a list, trimming whitespace."""
    if not value:
        return []
    return [addr.strip() for addr in value[0].split(",") if addr.strip()]


@swarm.command("new")
def deploy_new(
    ticker: str,
    name: str,
    rpc_url: str | None = None,
    members: Annotated[
        List[str],
        typer.Option(
            "--members",
            "-m",
            help="Comma-separated list of addresses",
            callback=csv_to_list,
        ),
    ] = [],
    mnemonic: MnemonicOption = None,
    account_name: AccountNameOption = None,
    account_index: int = 0,
):
    account_config = account_config_loader(mnemonic, account_name)
    account = account_config.get_account(account_index)

    model = ModelFactoryV1Config(name=name, ticker=ticker)

    web3_config = Web3Config(url=HttpUrl(url=rpc_url) if rpc_url else None)

    model_factory = ModelFactoryV1(model)
    deployment = model_factory.deploy(
        deployer=account, member_address=members, w3=web3_config.get_web3()
    )
    address = deployment.get_address()
    typer.echo(f"New federation deployed at {address}")
