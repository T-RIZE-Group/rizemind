from rizemind.exception.base_exception import RizemindException


class RizemindContractError(RizemindException):
    name: str
    error_args: dict[str, str]

    def __init__(self, name: str, error_args: dict[str, str]):
        super().__init__(
            code="contract_execution_error",
            message=f"Transaction reverted with {name}({error_args})",
        )
        self.name = name
        self.error_args = error_args

    def __str__(self):
        return f"Transaction reverted with {self.name}({self.error_args})"
