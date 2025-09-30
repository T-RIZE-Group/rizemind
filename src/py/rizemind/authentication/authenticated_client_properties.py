from eth_typing import ChecksumAddress
from flwr.server.client_proxy import ClientProxy
from pydantic import BaseModel

from rizemind.configuration.transform import from_properties, to_properties
from rizemind.exception.parse_exception import catch_parse_errors

AUTHENTICATED_CLIENT_PROPERTIES_PREFIX = "rizemind.authenticated_client_properties"


class AuthenticatedClientProperties(BaseModel):
    """The authenticated properties of a Flower client.

    Attributes:
        trainer_address: The trainer's wallet address.
    """

    trainer_address: ChecksumAddress

    def tag_client(self, client: ClientProxy):
        """Updates the `properties` dictionary of a client with its authentication properties.

        Args:
            client: The client to tag.
        """
        properties = to_properties(
            self.model_dump(), AUTHENTICATED_CLIENT_PROPERTIES_PREFIX
        )
        client.properties.update(properties)

    @catch_parse_errors
    @staticmethod
    def from_client(client: ClientProxy) -> "AuthenticatedClientProperties":
        """Constructs an AuthenticatedClientProperties instance from a client's properties.

        Args:
            client: The client from which to extract properties from.

        Returns:
            An instance of AuthenticatedClientProperties.
        """
        properties = client.properties
        return AuthenticatedClientProperties(
            **from_properties(properties)["rizemind"]["authenticated_client_properties"]
        )
