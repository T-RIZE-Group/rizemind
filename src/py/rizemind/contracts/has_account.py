from eth_account.signers.base import BaseAccount

from rizemind.exception.base_exception import RizemindException


class NoAccountConnected(RizemindException):
    def __init__(self):
        super().__init__("no_account_connected", "No Account Connected")


class HasAccount:
    _account: BaseAccount | None

    def __init__(self, *, account: BaseAccount | None = None):
        self._account = account

    def get_account(self) -> BaseAccount:
        if self._account is None:
            raise NoAccountConnected()
        return self._account

    def connect(self, account: BaseAccount):
        self._account = account
