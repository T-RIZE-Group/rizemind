from flwr.common import GetPropertiesIns, GetPropertiesRes
from flwr.common.typing import Code, Status
from pydantic import BaseModel

from rizemind.authentication.signatures.signature import Signature
from rizemind.configuration.transform import from_config, to_config
from rizemind.contracts.erc.erc5267.typings import EIP712DomainMinimal
from rizemind.exception.parse_exception import catch_parse_errors

TRAIN_AUTH_PREFIX = "rizemind.train_auth"


class TrainAuthInsConfig(BaseModel):
    domain: EIP712DomainMinimal
    round_id: int
    nonce: bytes


def prepare_train_auth_ins(
    *, round_id: int, nonce: bytes, domain: EIP712DomainMinimal
) -> GetPropertiesIns:
    config = TrainAuthInsConfig(domain=domain, round_id=round_id, nonce=nonce)
    return GetPropertiesIns(to_config(config.model_dump(), prefix=TRAIN_AUTH_PREFIX))


@catch_parse_errors
def parse_train_auth_ins(ins: GetPropertiesIns) -> TrainAuthInsConfig:
    config = from_config(ins.config)
    return TrainAuthInsConfig(**config["rizemind"]["train_auth"])


class RoundAuthResponseConfig(BaseModel):
    signature: Signature


def prepare_train_auth_res(signature: Signature) -> GetPropertiesRes:
    config = RoundAuthResponseConfig(signature=signature)
    return GetPropertiesRes(
        status=Status(code=Code.OK, message="auth signed"),
        properties=to_config(config.model_dump(), prefix=TRAIN_AUTH_PREFIX),
    )


@catch_parse_errors
def parse_train_auth_res(res: GetPropertiesRes):
    properties = from_config(res.properties)
    return RoundAuthResponseConfig(**properties["rizemind"]["train_auth"])
