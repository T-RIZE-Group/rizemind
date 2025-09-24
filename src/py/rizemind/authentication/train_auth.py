from flwr.common import GetPropertiesIns, GetPropertiesRes
from flwr.common.typing import Code, Status
from pydantic import BaseModel

from rizemind.authentication.signatures.signature import Signature
from rizemind.configuration.transform import from_config, to_config
from rizemind.contracts.erc.erc5267.typings import EIP712DomainMinimal
from rizemind.exception.parse_exception import catch_parse_errors

TRAIN_AUTH_PREFIX = "rizemind.train_auth"


class TrainAuthInsConfig(BaseModel):
    """Payload for a training-authorization request.

    Attributes:
        domain: Minimal EIP-712 domain used for signing.
        round_id: Federated learning round identifier.
        nonce: Anti-replay nonce associated with the round.
    """

    domain: EIP712DomainMinimal
    round_id: int
    nonce: bytes


def prepare_train_auth_ins(
    *, domain: EIP712DomainMinimal, round_id: int, nonce: bytes
) -> GetPropertiesIns:
    """Build a Flower `GetPropertiesIns` for training authorization.

    Args:
        domain: EIP-712 domain scoping the signature.
        round_id: Federated learning round identifier.
        nonce: Unique nonce for this request to prevent replay.

    Returns:
        A `GetPropertiesIns` whose properties are namespaced under
        `TRAIN_AUTH_PREFIX` value.
    """
    config = TrainAuthInsConfig(domain=domain, round_id=round_id, nonce=nonce)
    return GetPropertiesIns(to_config(config.model_dump(), prefix=TRAIN_AUTH_PREFIX))


@catch_parse_errors
def parse_train_auth_ins(ins: GetPropertiesIns) -> TrainAuthInsConfig:
    """Parse a training-authorization instruction payload.

    Args:
        ins: Incoming Flower `GetPropertiesIns` to parse.

    Returns:
        The parsed `TrainAuthInsConfig`.

    Raises:
        ParseException: If required fields are missing or invalid.
    """
    config = from_config(ins.config)
    return TrainAuthInsConfig(**config["rizemind"]["train_auth"])


class RoundAuthResponseConfig(BaseModel):
    """Payload for a training-authorization response.

    Attributes:
        signature: Signature produced by the client.
    """

    signature: Signature


def prepare_train_auth_res(signature: Signature) -> GetPropertiesRes:
    """Build a Flower `GetPropertiesRes` carrying the authorization signature.

    Args:
        signature: EIP-712 signature.

    Returns:
        A `GetPropertiesRes` with `OK` status and properties namespaced
        under `TRAIN_AUTH_PREFIX` value.
    """
    config = RoundAuthResponseConfig(signature=signature)
    return GetPropertiesRes(
        status=Status(code=Code.OK, message="auth signed"),
        properties=to_config(config.model_dump(), prefix=TRAIN_AUTH_PREFIX),
    )


@catch_parse_errors
def parse_train_auth_res(res: GetPropertiesRes):
    """Parse a training-authorization response payload.

    Args:
        res: Incoming Flower `GetPropertiesRes` to parse.

    Returns:
        The parsed `RoundAuthResponseConfig`.

    Raises:
        ParseException: If required fields are missing or invalid.
    """
    properties = from_config(res.properties)
    return RoundAuthResponseConfig(**properties["rizemind"]["train_auth"])
