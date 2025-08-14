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
    round_id: int
    domain: EIP712Domain
    swarm: SupportsEthAccountStrategy

    def __init__(self, round_id: int, swarm: SupportsEthAccountStrategy):
        self.round_id = round_id
        self.domain = swarm.get_eip712_domain()
        self.swarm = swarm

    def select(self, client: ClientProxy) -> bool:
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
