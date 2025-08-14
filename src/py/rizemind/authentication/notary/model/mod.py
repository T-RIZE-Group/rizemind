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
from rizemind.exception.base_exception import RizemindException
from rizemind.exception.parse_exception import ParseException
from rizemind.swarm.config import SwarmConfig
from rizemind.web3.config import Web3Config


class MissingConfigNotaryModError(RizemindException):
    def __init__(self):
        super().__init__(
            code="config_not_defined",
            message="config cannot be found in context state.",
        )


def model_notary_mod(
    msg: Message,
    ctxt: Context,
    call_next: ClientAppCallable,
) -> Message:
    """
    Weird behavior, but if you don't `call_next`, then ctxt won't contain values defined
    in the `client_fn`
    """
    if msg.metadata.message_type == MessageType.TRAIN:
        try:
            fit_ins = recorddict_to_fitins(msg.content, True)
            model_notary_config = parse_model_notary_config(fit_ins.config)
            swarm_config = SwarmConfig.from_context(
                ctxt, fallback_address=model_notary_config.domain.verifyingContract
            )
            web3_config = Web3Config.from_context(ctxt)
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

    reply = call_next(msg, ctxt)

    if msg.metadata.message_type == MessageType.TRAIN:
        try:
            account_config = AccountConfig.from_context(ctxt)
            fit_ins = recorddict_to_fitins(msg.content, True)
            model_notary_config = parse_model_notary_config(fit_ins.config)
            swarm_config = SwarmConfig.from_context(
                ctxt, fallback_address=model_notary_config.domain.verifyingContract
            )
            web3_config = Web3Config.from_context(ctxt)
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
