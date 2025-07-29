from eth_typing import Address, ChecksumAddress
from pydantic import BaseModel


class DeployedContract(BaseModel):
    address: ChecksumAddress

    def address_as_bytes(self) -> Address:
        return Address(bytes.fromhex(self.address[2:]))
