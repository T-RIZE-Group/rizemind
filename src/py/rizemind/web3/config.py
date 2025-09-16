from typing import Any

from flwr.common.context import Context
from pydantic import Field, HttpUrl, field_validator
from pydantic_core import Url
from rizemind.configuration.base_config import BaseConfig
from rizemind.web3.chains import RIZENET_TESTNET_CHAINID
from web3 import AsyncHTTPProvider, AsyncWeb3, HTTPProvider, Web3
from web3.main import BaseWeb3
from web3.middleware import ExtraDataToPOAMiddleware

poaChains = [RIZENET_TESTNET_CHAINID]

WEB3_CONFIG_STATE_KEY = "rizemind.web3"


class Web3Config(BaseConfig):
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

    def get_web3(self, *, web3_factory=Web3) -> Web3:
        w3 = web3_factory(self.web3_provider())
        self.inject_poa_middlewares(w3)
        return w3

    def web3_provider(self) -> HTTPProvider:
        url = self.get_rpc_url()
        return Web3.HTTPProvider(str(url))

    def get_rpc_url(self) -> str:
        url = self.url
        if url is None:
            url = "https://testnet.rizenet.io"
        return str(url)

    def inject_poa_middlewares(self, w3: BaseWeb3):
        if w3.eth.chain_id in poaChains:
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    def get_async_web3(self, *, web3_factory=AsyncWeb3) -> AsyncWeb3:
        w3 = web3_factory(self.async_web3_provider())
        self.inject_poa_middlewares(w3)
        return w3

    def async_web3_provider(self) -> AsyncHTTPProvider:
        url = self.get_rpc_url()
        return AsyncHTTPProvider(str(url))

    @staticmethod
    def from_context(context: Context) -> "Web3Config | None":
        if WEB3_CONFIG_STATE_KEY in context.state.config_records:
            records: Any = context.state.config_records[WEB3_CONFIG_STATE_KEY]
            return Web3Config(**records)
        return None
