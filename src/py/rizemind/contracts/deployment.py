from eth_typing import HexAddress, Address
from pydantic import BaseModel


class DeployedContract(BaseModel):
    address: HexAddress
    abi: list[dict]

    def address_as_bytes(self) -> Address:
        return Address(bytes.fromhex(self.address[2:]))
