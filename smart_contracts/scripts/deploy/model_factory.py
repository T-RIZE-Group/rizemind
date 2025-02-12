import click
from ape import project, networks
from ape.logging import logger
from ape.cli import network_option, account_option
from output import save_contract_data

@click.command()
@network_option() 
@account_option()
def cli(ecosystem, network, provider, account):
    logger.info(f"Deploying on {ecosystem.name}:{network.name}:{provider.name}, Account {account}")
    # Connect to the network
    with networks.parse_network_choice(f"{ecosystem.name}:{network.name}:{provider.name}") as provider:
      model = project.ModelRegistryV1.deploy(sender=account)
      save_contract_data(model, output_folder=f'output/{network.name}')
      factory = project.ModelRegistryFactory.deploy(model.address, sender=account)
      if network.name != "local":
        project.deployments.track(factory)
      logger.info(f"Factory deployed at {factory.address}")
      save_contract_data(factory, output_folder=f'output/{network.name}')
