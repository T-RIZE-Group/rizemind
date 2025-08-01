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


class AuthenticationError(RizemindException):
    def __init__(self):
        super().__init__(
            code="account_config_not_defined",
            message="AccountConfig cannot be found in context state.",
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
            raise AuthenticationError()
        account = account_config.get_account()
        get_properties_ins = recorddict_to_getpropertiesins(msg.content)
        train_auth_ins = parse_train_auth_ins(get_properties_ins)
        # TODO: validate domain agaisnt configure domain
        signature = sign_auth_message(
            account=account,
            round=train_auth_ins.round_id,
            nonce=train_auth_ins.nonce,
            domain=train_auth_ins.domain,
        )
        res = prepare_train_auth_res(signature)
        return Message(getpropertiesres_to_recorddict(res), reply_to=msg)

    return reply
