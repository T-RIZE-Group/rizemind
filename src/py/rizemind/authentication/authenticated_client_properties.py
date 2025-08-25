from eth_typing import ChecksumAddress
from flwr.server.client_proxy import ClientProxy
from pydantic import BaseModel

from rizemind.configuration.transform import from_properties, to_properties
from rizemind.exception.parse_exception import catch_parse_errors

AUTHENTICATED_CLIENT_PROPERTIES_PREFIX = "rizemind.authenticated_client_properties"


class AuthenticatedClientProperties(BaseModel):
    trainer_address: ChecksumAddress

    def tag_client(self, client: ClientProxy):
        properties = to_properties(
            self.model_dump(), AUTHENTICATED_CLIENT_PROPERTIES_PREFIX
        )
        client.properties.update(properties)

    @catch_parse_errors
    @staticmethod
    def from_client(client: ClientProxy) -> "AuthenticatedClientProperties":
        properties = client.properties
        return AuthenticatedClientProperties(
            **from_properties(properties)["rizemind"]["authenticated_client_properties"]
        )
