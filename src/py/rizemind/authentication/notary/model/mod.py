import logging

from flwr.client.typing import ClientAppCallable
from flwr.common import (
    Context,
    Message,
    log,
)
from flwr.common.constant import MessageType
from flwr.common.recorddict_compat import (
    fitres_to_recorddict,
    recorddict_to_fitins,
    recorddict_to_fitres,
)

from rizemind.authentication.config import AccountConfig
from rizemind.authentication.notary.model.config import (
    parse_model_notary_config,
    prepare_model_notary_config,
)
from rizemind.authentication.notary.model.model_signature import (
    hash_parameters,
    recover_model_signer,
    sign_parameters_model,
)
from rizemind.configuration.transform import concat
from rizemind.exception import ParseException, RizemindException
from rizemind.swarm.config import SwarmConfig
from rizemind.web3 import Web3Config


class MissingConfigNotaryModError(RizemindException):
    """Raised when the configuration for the notary mod is missing from the context.

    This error indicates that the notary mod was initialized or executed without its
    required configuration being available in the context.
    """

    def __init__(self):
        super().__init__(
            code="config_not_defined",
            message="config cannot be found in context state.",
        )


def model_notary_mod(
    msg: Message,
    ctx: Context,
    call_next: ClientAppCallable,
) -> Message:
    """Flower client mod for model parameter authentication and signing.

    This mod intercepts TRAIN messages to verify incoming model parameters
    and sign outgoing model parameters using EIP-712 signatures. It ensures that
    models can be authenticated given the swarm's aggregator registry.

    Args:
        msg: The Flower message containing the model parameters and metadata.
        ctx: The Flower context containing configuration for authentication,
        including SwarmConfig, Web3Config, and AccountConfig.
        call_next: The next callable in the chain to process the message.

    Raises:
        MissingConfigNotaryModError: When required configuration (SwarmConfig,
        Web3Config, or AccountConfig) is not found in the context.
        RizemindException: When the recovered signer is not a valid aggregator
        for the given round.

    Returns:
        The processed message with signed parameters for TRAIN messages.
            For non-TRAIN messages, returns the unmodified reply from call_next.

    Notes:
        - On incoming TRAIN messages, verifies the model signature and checks
            if the signer is a valid aggregator. This signer must be the aggregator
            sending the parameters.
        - On outgoing TRAIN responses, signs the model parameters with the
            client's account and includes the signature in the metrics.
        - Logs warnings when verification fails and errors when signing fails.
        - Non-TRAIN messages pass through without modification.
    """
    # Weird behavior, but if you don't `call_next`,
    # then ctx won't contain values defined in the `client_fn`
    if msg.metadata.message_type == MessageType.TRAIN:
        try:
            fit_ins = recorddict_to_fitins(msg.content, True)
            model_notary_config = parse_model_notary_config(fit_ins.config)
            swarm_config = SwarmConfig.from_context(
                ctx, fallback_address=model_notary_config.domain.verifyingContract
            )
            web3_config = Web3Config.from_context(ctx)
            if swarm_config is None or web3_config is None:
                raise MissingConfigNotaryModError()
            else:
                swarm = swarm_config.get(w3=web3_config.get_web3())
                domain = swarm.get_eip712_domain()
                model_signer = recover_model_signer(
                    model=fit_ins.parameters,
                    round=model_notary_config.round_id,
                    domain=domain,
                    signature=model_notary_config.signature,
                )
                if not swarm.is_aggregator(model_signer, model_notary_config.round_id):
                    raise RizemindException(
                        code="not_an_aggregator",
                        message=f"{model_signer} is not an aggregator",
                    )
        except (ParseException, MissingConfigNotaryModError):
            log(
                level=logging.WARN,
                msg="Impossible to verify parameters authenticity: Cannot find swarm config or web3 config",
            )

    reply = call_next(msg, ctx)

    if msg.metadata.message_type == MessageType.TRAIN:
        try:
            account_config = AccountConfig.from_context(ctx)
            fit_ins = recorddict_to_fitins(msg.content, True)
            model_notary_config = parse_model_notary_config(fit_ins.config)
            swarm_config = SwarmConfig.from_context(
                ctx, fallback_address=model_notary_config.domain.verifyingContract
            )
            web3_config = Web3Config.from_context(ctx)
            if account_config is None or swarm_config is None or web3_config is None:
                raise MissingConfigNotaryModError()
            else:
                swarm = swarm_config.get(w3=web3_config.get_web3())
                domain = swarm.get_eip712_domain()
                account = account_config.get_account()
                fit_res = recorddict_to_fitres(reply.content, False)
                signature = sign_parameters_model(
                    account=account,
                    domain=domain,
                    parameters=fit_res.parameters,
                    round=model_notary_config.round_id,
                )

                notary_config = prepare_model_notary_config(
                    round_id=model_notary_config.round_id,
                    domain=domain,
                    signature=signature,
                    model_hash=hash_parameters(fit_res.parameters),
                )
                fit_res.metrics = concat(fit_res.metrics, notary_config)
                reply.content = fitres_to_recorddict(fit_res, False)

        except (ParseException, MissingConfigNotaryModError):
            log(
                level=logging.ERROR,
                msg="Impossible to sign parameters: Cannot find required configs",
            )
    return reply
