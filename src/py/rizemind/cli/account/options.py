from typing import Annotated, Optional

import typer

AccountNameOption = Annotated[
    Optional[str],
    typer.Option(
        "--account-name",
        "-a",
        help="Name of the account.",
        show_default=False,
    ),
]


MnemonicOption = Annotated[
    Optional[str],
    typer.Option(
        "--mnemonic",
        "-m",
        help="BIP-39 seed phrase",
        show_default=False,
    ),
]
