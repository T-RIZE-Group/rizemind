from typing import Any

from flwr.common.context import Context
from pydantic import Field, HttpUrl, field_validator
from pydantic_core import Url
from web3 import AsyncHTTPProvider, AsyncWeb3, HTTPProvider, Web3
from web3.middleware import ExtraDataToPOAMiddleware

from rizemind.configuration.base_config import BaseConfig
from rizemind.configuration.transform import unflatten
from rizemind.web3.chains import RIZENET_TESTNET_CHAINID
from rizemind.web3.middlewares.errors import RizemindErrorsMiddleware

poaChains = [RIZENET_TESTNET_CHAINID]

WEB3_CONFIG_STATE_KEY = "rizemind.web3"


class Web3Config(BaseConfig):
    """Configuration for Web3 blockchain connections.

    Manages Web3 provider configuration including URL validation, middleware
    injection for Proof-of-Authority chains, and Web3 instance creation.

    Attributes:
        url: The HTTP provider URL for blockchain connectivity. If None,
            defaults to RizeNet testnet URL.
    """

    url: HttpUrl | None = Field(..., description="The HTTP provider URL")

    @field_validator("url", mode="before")
    @classmethod
    def coerce_url(cls, value: Any) -> Any:
        """Validates and coerces URL values to HttpUrl format.

        Handles various input types and converts them to the appropriate
        HttpUrl format for Pydantic validation.

        Args:
            value: The URL value to validate. Can be None, string, HttpUrl, or Url.

        Returns:
            The validated HttpUrl object or None.

        Raises:
            TypeError: If the value type is not supported.
            ValidationError: If the URL string is invalid (raised by HttpUrl).
        """
        # Let Pydantic handle None or already-parsed HttpUrl
        if value is None or isinstance(value, HttpUrl | Url):
            return value
        if isinstance(value, str):
            return HttpUrl(value)  # will raise if invalid
        raise TypeError("url must be a string, HttpUrl, or None")

    def get_web3(self, *, web3_factory=Web3) -> Web3:
        """Creates a configured Web3 instance.

        Initializes a Web3 instance with the configured HTTP provider and
        automatically injects Proof-of-Authority middleware for supported chains.

        Args:
            web3_factory: Factory function for creating Web3 instances.
                Defaults to the standard Web3 constructor.

        Returns:
            A configured Web3 instance ready for blockchain interaction.
        """
        w3 = web3_factory(self.web3_provider())
        if w3.eth.chain_id in poaChains:
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        w3.middleware_onion.add(
            RizemindErrorsMiddleware, name="rizemind_error_middleware"
        )
        return w3

    def web3_provider(self) -> HTTPProvider:
        """Creates an HTTP provider for Web3 connections.

        Constructs an HTTPProvider using the configured URL or defaults
        to the RizeNet testnet if no URL is specified.

        Returns:
            An HTTPProvider instance configured with the appropriate endpoint URL.
        """
        url = self.get_rpc_url()
        return Web3.HTTPProvider(str(url))

    def get_rpc_url(self) -> str:
        url = self.url
        if url is None:
            url = "https://testnet.rizenet.io"
        return str(url)

    async def get_async_web3(self, *, web3_factory=AsyncWeb3) -> AsyncWeb3:
        w3 = web3_factory(self.async_web3_provider())
        chain_id = await w3.eth.chain_id
        if chain_id in poaChains:
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return w3

    def async_web3_provider(self) -> AsyncHTTPProvider:
        url = self.get_rpc_url()
        return AsyncHTTPProvider(str(url))

    def store_in_context(self, context: Context) -> None:
        self._store_in_context(context, WEB3_CONFIG_STATE_KEY)

    @staticmethod
    def from_context(context: Context) -> "Web3Config | None":
        """Creates a Web3Config instance from a Flower context.

        Extracts Web3 configuration from the provided Flower context state
        and constructs a Web3Config instance if the configuration exists.

        Args:
            context: The Flower context containing configuration state.

        Returns:
            A Web3Config instance if configuration is found in the context,
            otherwise None.
        """
        if WEB3_CONFIG_STATE_KEY in context.state.config_records:
            records: Any = context.state.config_records[WEB3_CONFIG_STATE_KEY]
            return Web3Config(**unflatten(records))
        return None
