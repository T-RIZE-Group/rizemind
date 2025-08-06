from typing import cast

from eth_typing import ChecksumAddress
from flwr.client.typing import ClientAppCallable
from flwr.common import (
    Context,
    Message,
)
from flwr.common.constant import MessageTypeLegacy
from flwr.common.recorddict_compat import (
    getpropertiesres_to_recorddict,
    recorddict_to_getpropertiesins,
)

from rizemind.authentication.config import AccountConfig
from rizemind.authentication.signatures.auth import sign_auth_message
from rizemind.authentication.train_auth import (
    parse_train_auth_ins,
    prepare_train_auth_res,
)
from rizemind.exception.base_exception import RizemindException
from rizemind.exception.parse_exception import ParseException
from rizemind.swarm.config import SwarmConfig


class NoAccountAuthenticationModError(RizemindException):
    def __init__(self):
        super().__init__(
            code="no_account_config",
            message="AccountConfig cannot be found in context state.",
        )


class NoSwarmAuthenticationModError(RizemindException):
    def __init__(self):
        super().__init__(
            code="no_swarm_config",
            message="SwarmConfig cannot be found in context state.",
        )


class WrongSwarmAuthenticationModError(RizemindException):
    def __init__(self, expected: ChecksumAddress | None, received: ChecksumAddress):
        super().__init__(
            code="wrong_swarm_domain",
            message=f"Swarm domain {received} does not match configured domain {expected}.",
        )


def authentication_mod(
    msg: Message,
    ctxt: Context,
    call_next: ClientAppCallable,
) -> Message:
    """
    Weird behavior, but if you don't `call_next`, then ctxt won't contain values defined
    in the `client_fn`
    """
    reply = call_next(msg, ctxt)

    if msg.metadata.message_type == MessageTypeLegacy.GET_PROPERTIES:
        account_config = AccountConfig.from_context(ctxt)
        if account_config is None:
            raise NoAccountAuthenticationModError()
        account = account_config.get_account()
        get_properties_ins = recorddict_to_getpropertiesins(msg.content)
        try:
            train_auth_ins = parse_train_auth_ins(get_properties_ins)
            swarm_config = SwarmConfig.from_context(
                ctxt, fallback_address=train_auth_ins.domain.verifyingContract
            )
            if swarm_config is None:
                raise NoSwarmAuthenticationModError()
            if swarm_config.address != train_auth_ins.domain.verifyingContract:
                raise WrongSwarmAuthenticationModError(
                    cast(ChecksumAddress, swarm_config.address),
                    train_auth_ins.domain.verifyingContract,
                )

            signature = sign_auth_message(
                account=account,
                round=train_auth_ins.round_id,
                nonce=train_auth_ins.nonce,
                domain=train_auth_ins.domain,
            )
            res = prepare_train_auth_res(signature)
            return Message(getpropertiesres_to_recorddict(res), reply_to=msg)
        except ParseException:
            pass

    return reply
