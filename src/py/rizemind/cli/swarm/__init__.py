from pathlib import Path
from typing import Annotated, cast

import typer
from eth_typing import ChecksumAddress
from pydantic import HttpUrl
from web3 import Web3

from rizemind.cli.account.loader import account_config_loader
from rizemind.cli.account.options import AccountNameOption, MnemonicOption
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import SwarmV1FactoryConfig
from rizemind.swarm.certificate.certificate import Certificate, CompressedCertificate
from rizemind.swarm.config import SwarmConfig
from rizemind.web3.config import Web3Config

swarm = typer.Typer(help="Federation management commands")


def split_addresses(
    value: list[str],
) -> list[ChecksumAddress]:
    """Split a list of comma-separated strings into a flat list, trimming whitespace."""
    if not value:
        return []

    return [
        Web3.to_checksum_address(item.strip())
        for part in value
        for item in part.split(",")
        if item.strip()
    ]


@swarm.command("new")
def deploy_new(
    ticker: str,
    name: str,
    rpc_url: str | None = None,
    members: Annotated[
        list[str],
        typer.Option("--members", "-m", help="Comma-separated list of addresses"),
    ] = [],
    mnemonic: MnemonicOption = None,
    account_name: AccountNameOption = None,
    account_index: int = 0,
):
    account_config = account_config_loader(mnemonic, account_name)
    account = account_config.get_account(account_index)
    factory_config = SwarmV1FactoryConfig(
        name=name,
        ticker=ticker,
    )
    swarm_config = SwarmConfig(factory_v1=factory_config, address=None)

    web3_config = Web3Config(url=cast(HttpUrl | None, rpc_url))
    trainers = split_addresses(members)
    deployment = swarm_config.deploy(
        deployer=account, trainers=trainers, w3=web3_config.get_web3()
    )
    address = deployment.address
    typer.echo(f"New Swarm deployed at {address}")


@swarm.command("set-cert")
def set_cert(
    cert_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            resolve_path=True,
            help="PEM or DER certificate file",
        ),
    ],
    address: Annotated[
        str,
        typer.Option(
            "--address",
            "-a",
            help="The address of the swarm",
            callback=Web3.to_checksum_address,
        ),
    ],
    id: Annotated[
        str,
        typer.Option(
            "--id",
            help="The certificate ID",
        ),
    ],
    rpc_url: Annotated[
        str | None,
        typer.Option(
            "--rpc-url",
            help="The RPC url for the blockchain",
        ),
    ] = None,
    mnemonic: MnemonicOption = None,
    account_name: AccountNameOption = None,
    account_index: int = 0,
) -> None:
    """
    Read *CERT_PATH* certificate bytes
    and writes it onchain
    """
    cert: Certificate = Certificate.from_path(cert_path)

    cc: CompressedCertificate = cert.get_compressed_bytes()
    payload: bytes = cc.data
    algo_name: str = cc.algorithm.value

    swarm_config = SwarmConfig(address=address)
    web3_config = Web3Config(url=cast(HttpUrl | None, rpc_url))
    w3 = web3_config.get_web3()
    account_config = account_config_loader(mnemonic, account_name)
    account = account_config.get_account(account_index)
    swarm = swarm_config.get(w3=w3, account=account)
    success = swarm.set_certificate(id, cert)
    if success:
        typer.echo(f"Store [{algo_name}] {len(payload)} bytes onchain")
    else:
        typer.echo("Failed to set certificate", err=True)


@swarm.command("download-cert")
def download_cert(
    cert_path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            writable=True,
            resolve_path=True,
            help="PEM or DER certificate file",
        ),
    ],
    address: Annotated[
        str,
        typer.Option(
            "--address",
            "-a",
            help="The address of the swarm",
            callback=Web3.to_checksum_address,
        ),
    ],
    id: Annotated[
        str,
        typer.Option(
            "--id",
            help="The certificate ID",
        ),
    ],
    rpc_url: Annotated[
        str | None,
        typer.Option(
            "--rpc-url",
            help="The RPC url for the blockchain",
        ),
    ] = None,
) -> None:
    """
    Stores certificates locally at *CERT_PATH*
    """

    swarm_config = SwarmConfig(address=address)
    web3_config = Web3Config(url=cast(HttpUrl | None, rpc_url))
    w3 = web3_config.get_web3()
    swarm = swarm_config.get(w3=w3)
    cert = swarm.get_certificate(id)
    if cert is None:
        typer.echo("No certificate found", err=True)
    else:
        cert.store(cert_path)
        typer.echo(f"Stored certificate at {cert_path}")
