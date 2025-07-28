from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator
from pydantic_core import Url
from rizemind.web3.chains import RIZENET_TESTNET_CHAINID
from web3 import HTTPProvider, Web3
from web3.middleware import ExtraDataToPOAMiddleware

poaChains = [RIZENET_TESTNET_CHAINID]


class Web3Config(BaseModel):
    url: HttpUrl | None = Field(..., description="The HTTP provider URL")

    @field_validator("url", mode="before")
    @classmethod
    def coerce_url(cls, value: Any) -> Any:
        # Let Pydantic handle None or already-parsed HttpUrl
        if value is None or isinstance(value, HttpUrl | Url):
            return value
        if isinstance(value, str):
            return HttpUrl(value)  # will raise if invalid
        raise TypeError("url must be a string, HttpUrl, or None")

    def get_web3(self) -> Web3:
        w3 = Web3(self.web3_provider())
        if w3.eth.chain_id in poaChains:
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return w3

    def web3_provider(self) -> HTTPProvider:
        url = self.url
        if url is None:
            url = "https://testnet.rizenet.io"
        return Web3.HTTPProvider(str(url))
