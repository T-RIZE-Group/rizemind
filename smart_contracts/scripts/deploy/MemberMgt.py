import click
from ape import project, networks
from ape.logging import logger
from ape.cli import network_option, account_option


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
        # Deploy the contract
        # address[] memory initialMembers, uint256 initialThreshold
        # (membersaccount array, threshold value, sender=accounts)  set w.r.t constructor of smart contract
        contract = project.MemberManagement.deploy([account], 1, sender=account)
        if network.name != "local":
            project.deployments.track(contract)
        logger.info(f"Contract deployed at {contract.address}")
