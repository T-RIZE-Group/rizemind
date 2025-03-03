from pydantic import BaseModel, Field, HttpUrl
from rizemind.web3.chains import RIZENET_TESTNET_CHAINID
from web3 import HTTPProvider
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

poaChains = [RIZENET_TESTNET_CHAINID]


class Web3Config(BaseModel):
    url: HttpUrl = Field(..., description="The HTTP provider URL")

    def get_web3(self) -> Web3:
        w3 = Web3(self.web3_provider())
        if w3.eth.chain_id in poaChains:
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return w3

    def web3_provider(self) -> HTTPProvider:
        return Web3.HTTPProvider(str(self.url))
