from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from rizemind.authentication.authenticated_client_properties import (
    AuthenticatedClientProperties,
)
from rizemind.authentication.typing import SupportsEthAccountStrategy
from rizemind.contracts.erc.erc5267.typings import EIP712Domain
from rizemind.exception.parse_exception import ParseException


class CanTrainCriterion(Criterion):
    """Flower criterion to select clients that can train.

    This criterion implements a check to authenticate clients using Ethereum signatures.
    It verifies if a client is authorized to participate in a specific training round.

    Attributes:
        round_id: The identifier of the current training round.
        domain: The EIP-712 domain for signing authentication messages.
        swarm: The protocol that provides the EIP-712 domain and
        verifies training permissions.
    """

    round_id: int
    domain: EIP712Domain
    swarm: SupportsEthAccountStrategy

    def __init__(self, round_id: int, swarm: SupportsEthAccountStrategy):
        """Initializes the CanTrainCriterion.

        Args:
            round_id: The ID of the current training round.
            swarm: The protocol that provides the EIP-712 domain and
            verifies training permissions.
        """
        self.round_id = round_id
        self.domain = swarm.get_eip712_domain()
        self.swarm = swarm

    def select(self, client: ClientProxy) -> bool:
        """Selects a client for training based on authentication.

        This method reads the client's properties and recover's its address
        to determine whether it can participate in the training.

        Args:
            client: The client proxy to evaluate for selection.

        Returns:
            True if the client is authenticated and authorized to train,
            False otherwise.
        """
        try:
            signer = AuthenticatedClientProperties.from_client(client).trainer_address
            return self.swarm.can_train(signer, self.round_id)
        except (ParseException, ValueError):
            return False
