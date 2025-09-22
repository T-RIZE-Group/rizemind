import logging

from flwr.client.typing import ClientAppCallable
from flwr.common import (
    Context,
    Message,
    log,
)
from flwr.common.constant import MessageType
from flwr.common.recorddict_compat import (
    recorddict_to_fitres,
)
from hexbytes import HexBytes
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.notary.model.config import (
    parse_model_notary_config,
)
from rizemind.contracts.base_contract import RizemindContractException
from rizemind.exception.base_exception import RizemindException
from rizemind.exception.parse_exception import ParseException
from rizemind.swarm.config import SwarmConfig
from rizemind.swarm.swarm import SwarmException
from rizemind.web3.config import Web3Config


class MissingConfigRegisterContributionModError(RizemindException):
    def __init__(self):
        super().__init__(
            code="config_not_defined",
            message="config cannot be found in context state.",
        )


def register_contribution_mod(
    msg: Message,
    ctxt: Context,
    call_next: ClientAppCallable,
) -> Message:
    """
    Weird behavior, but if you don't `call_next`, then ctxt won't contain values defined
    in the `client_fn`
    """
    reply = call_next(msg, ctxt)

    if msg.metadata.message_type == MessageType.TRAIN:
        try:
            account_config = AccountConfig.from_context(ctxt)
            web3_config = Web3Config.from_context(ctxt)
            if account_config is None or web3_config is None:
                raise MissingConfigRegisterContributionModError()
            else:
                fit_res = recorddict_to_fitres(reply.content, True)
                model_notary_config = parse_model_notary_config(fit_res.metrics)
                swarm_config = SwarmConfig.from_context(
                    ctxt, fallback_address=model_notary_config.domain.verifyingContract
                )
                if swarm_config is not None:
                    account = account_config.get_account()
                    swarm = swarm_config.get(w3=web3_config.get_web3(), account=account)
                    swarm.register_round_contribution(
                        model_notary_config.round_id,
                        HexBytes(model_notary_config.model_hash),
                    )
        except ParseException as e:
            log(
                level=logging.ERROR,
                msg=f"Impossible to register round contribution: {e}. Make sure `register_contribution_mod` is before the `model_notary_mod` in the ClientApp.mods",
            )
        except MissingConfigRegisterContributionModError:
            log(
                level=logging.ERROR,
                msg="Impossible to register round contribution: Cannot find required configs",
            )
        except (RizemindContractException, SwarmException) as e:
            log(
                level=logging.ERROR,
                msg=f"Impossible to register round contribution: {e.message}. ",
            )
        except Exception as e:
            log(
                level=logging.ERROR,
                msg=f"Impossible to register round contribution: {e}",
            )
    return reply
