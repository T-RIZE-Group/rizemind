import click
from ape import project, networks
from ape.logging import logger
from ape.cli import network_option, account_option, AccountAliasPromptChoice

@click.command()
@network_option()
@click.option(
    "--contract"
)
@click.option(
    "--member",
    type=AccountAliasPromptChoice()
)
@account_option()
def cli(ecosystem, network, provider, member, account, contract):
  # Connect to the network
  with networks.parse_network_choice(f"{ecosystem.name}:{network.name}:{provider.name}") as provider:
    logger.info(f"Adding {member} on {ecosystem.name}:{network.name}:{provider.name}:{contract}")
    contract = project.MemberManagement.at(contract)
    contract.proposeAddMember(member, sender=account)
    