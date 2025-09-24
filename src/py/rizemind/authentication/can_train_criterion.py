import os

from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from rizemind.authentication.signatures.auth import recover_auth_signer
from rizemind.authentication.train_auth import (
    parse_train_auth_res,
    prepare_train_auth_ins,
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
        nonce = os.urandom(32)
        ins = prepare_train_auth_ins(
            round_id=self.round_id, nonce=nonce, domain=self.domain
        )

        try:
            res = client.get_properties(ins, timeout=60, group_id=self.round_id)
            auth = parse_train_auth_res(res)
            signer = recover_auth_signer(
                round=self.round_id,
                nonce=nonce,
                domain=self.domain,
                signature=auth.signature,
            )
            return self.swarm.can_train(signer, self.round_id)
        except (ParseException, ValueError):
            return False
