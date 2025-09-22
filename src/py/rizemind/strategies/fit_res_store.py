from typing import Protocol

from eth_typing import ChecksumAddress
from flwr.common import FitRes


class SupportsFitResStore(Protocol):
    def insert(self, client_id: ChecksumAddress, fit_res: FitRes) -> None: ...
    def get(self, client_id: ChecksumAddress) -> FitRes: ...
    def clear(self) -> None: ...
    def items(self) -> list[tuple[ChecksumAddress, FitRes]]: ...


class InMemoryFitResStore(SupportsFitResStore):
    fit_res_store: dict[ChecksumAddress, FitRes]

    def __init__(self) -> None:
        self.fit_res_store = {}

    def insert(self, client_id: ChecksumAddress, fit_res: FitRes) -> None:
        self.fit_res_store[client_id] = fit_res

    def get(self, client_id: ChecksumAddress) -> FitRes:
        return self.fit_res_store[client_id]

    def clear(self) -> None:
        self.fit_res_store = {}

    def items(self) -> list[tuple[ChecksumAddress, FitRes]]:
        return list(self.fit_res_store.items())
