from eth_typing import Address, HexAddress
from pydantic import BaseModel


class DeployedContract(BaseModel):
    address: HexAddress

    def address_as_bytes(self) -> Address:
        return Address(bytes.fromhex(self.address[2:]))
