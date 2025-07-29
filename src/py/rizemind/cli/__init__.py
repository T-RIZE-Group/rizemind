import typer

from .account import account
from .swarm import swarm

rzmnd = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
rzmnd.add_typer(account, name="account")
rzmnd.add_typer(swarm, name="swarm")

if __name__ == "__main__":  # allows `python cli.py` during dev
    rzmnd()
