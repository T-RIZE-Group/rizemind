import click
from ape import project, networks
from ape.logging import logger
from ape.cli import network_option, account_option
from output import save_contract_data


@click.command()
@network_option()
@account_option()
def cli(ecosystem, network, provider, account):
    logger.info(
        f"Deploying on {ecosystem.name}:{network.name}:{provider.name}, Account {account}"
    )
    # Connect to the network
    with networks.parse_network_choice(
        f"{ecosystem.name}:{network.name}:{provider.name}"
    ) as provider:
        model = project.ModelRegistryV1.deploy(sender=account)
        version = model.eip712Domain()
        save_contract_data(
            model,
            output_folder=f"output/{provider.chain_id}",
            version=version["version"],
        )
        tx = project.ModelRegistryFactory.deployments[-1].updateImplementation(
            model.address, sender=account
        )
