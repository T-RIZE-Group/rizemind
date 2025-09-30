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
from rizemind.exception import ParseException, RizemindException
from rizemind.swarm.config import SwarmConfig


class NoAccountAuthenticationModError(RizemindException):
    """No account configuration is available in the context state."""

    def __init__(self):
        super().__init__(
            code="no_account_config",
            message="AccountConfig cannot be found in context state.",
        )


class NoSwarmAuthenticationModError(RizemindException):
    """No swarm configuration is available in the context state."""

    def __init__(self):
        super().__init__(
            code="no_swarm_config",
            message="SwarmConfig cannot be found in context state.",
        )


class WrongSwarmAuthenticationModError(RizemindException):
    """The requested swarm domain does not match the configured domain."""

    def __init__(self, expected: ChecksumAddress | None, received: ChecksumAddress):
        super().__init__(
            code="wrong_swarm_domain",
            message=f"Swarm domain {received} does not match configured domain {expected}.",
        )


def authentication_mod(
    msg: Message,
    ctx: Context,
    call_next: ClientAppCallable,
) -> Message:
    """Handle authentication for GET_PROPERTIES messages.

    Invokes the next callable to populate the context, then, for messages
    carrying training-auth instructions, verifies the swarm domain and
    returns a signed authentication response.

    Args:
        msg: Incoming message to process.
        ctx: Flower context carrying client state.
        call_next: Next app callable to delegate to.

    Returns:
        The resulting message. If authentication applies, this is a
        GET_PROPERTIES response containing the signature; otherwise, the
        delegated reply.

    Raises:
        NoAccountAuthenticationModError: Account configuration is missing
            from context.
        NoSwarmAuthenticationModError: Swarm configuration is missing
            from context.
        WrongSwarmAuthenticationModError: Request domain differs from the
            configured swarm domain.
    """
    # Weird behavior, but if you don't `call_next`,
    # then ctx won't contain values defined
    # in the `client_fn`
    reply = call_next(msg, ctx)

    if msg.metadata.message_type == MessageTypeLegacy.GET_PROPERTIES:
        account_config = AccountConfig.from_context(ctx)
        if account_config is None:
            raise NoAccountAuthenticationModError()
        account = account_config.get_account()
        get_properties_ins = recorddict_to_getpropertiesins(msg.content)
        try:
            train_auth_ins = parse_train_auth_ins(get_properties_ins)
            swarm_config = SwarmConfig.from_context(
                ctx, fallback_address=train_auth_ins.domain.verifyingContract
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
