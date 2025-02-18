from pydantic import BaseModel, Field, HttpUrl
from web3 import HTTPProvider
from web3 import Web3

class Web3Config(BaseModel):
    url: HttpUrl = Field(..., description="The HTTP provider URL")
    
    def get_web3(self) -> Web3:
        return Web3(self.web3_provider())

    def web3_provider(self) -> HTTPProvider:
        return Web3.HTTPProvider(self.url)