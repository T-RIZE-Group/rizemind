import typer
from .account import account

rzmnd = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
rzmnd.add_typer(account, name="account")

if __name__ == "__main__":  # allows `python cli.py` during dev
    rzmnd()
