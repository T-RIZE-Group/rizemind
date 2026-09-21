from eth_typing import Address
from pydantic import BaseModel

from rizemind.configuration.validators import EthereumAddress


class DeployedContract(BaseModel):
    """A contract deployment, addressed on a specific chain.

    The address is validated and normalized to EIP-55 on construction. Addresses
    reach this model straight from configuration — a `factory_deployments`
    override in an example's `pyproject.toml`, say, where `TomlConfig` leaves an
    unset `$VAR` as a literal — so an unchecked string would only fail much
    later, inside `address_as_bytes`.
    """

    address: EthereumAddress

    def address_as_bytes(self) -> Address:
        return Address(bytes.fromhex(self.address[2:]))
