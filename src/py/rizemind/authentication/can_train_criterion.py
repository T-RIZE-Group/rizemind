from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from rizemind.authentication.authenticated_client_properties import (
    AuthenticatedClientProperties,
)
from rizemind.authentication.typing import SupportsEthAccountStrategy
from rizemind.contracts.erc.erc5267.typings import EIP712Domain
from rizemind.exception.parse_exception import ParseException


class CanTrainCriterion(Criterion):
    round_id: int
    domain: EIP712Domain
    swarm: SupportsEthAccountStrategy

    def __init__(self, round_id: int, swarm: SupportsEthAccountStrategy):
        self.round_id = round_id
        self.domain = swarm.get_eip712_domain()
        self.swarm = swarm

    def select(self, client: ClientProxy) -> bool:
        try:
            signer = AuthenticatedClientProperties.from_client(client).trainer_address
            return self.swarm.can_train(signer, self.round_id)
        except (ParseException, ValueError):
            return False
