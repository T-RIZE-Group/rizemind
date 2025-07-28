from typing import Annotated

import typer

AccountNameOption = Annotated[
    str | None,
    typer.Option(
        "--account-name",
        "-a",
        help="Name of the account.",
        show_default=False,
    ),
]


MnemonicOption = Annotated[
    str | None,
    typer.Option(
        "--mnemonic",
        "-m",
        help="BIP-39 seed phrase",
        show_default=False,
    ),
]
