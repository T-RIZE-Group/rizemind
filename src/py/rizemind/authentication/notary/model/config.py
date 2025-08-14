from typing import Any

from pydantic import BaseModel
from rizemind.authentication.signatures.signature import Signature
from rizemind.configuration.transform import from_config, to_config
from rizemind.contracts.erc.erc5267.typings import EIP712DomainMinimal
from rizemind.exception.parse_exception import catch_parse_errors

MODEL_NOTARY_PREFIX = "rizemind.notary.model"


class ModelNotaryConfig(BaseModel):
    domain: EIP712DomainMinimal
    round_id: int
    model_hash: bytes
    signature: Signature


def prepare_model_notary_config(
    *,
    round_id: int,
    domain: EIP712DomainMinimal,
    signature: Signature,
    model_hash: bytes,
):
    config = ModelNotaryConfig(
        domain=domain,
        round_id=round_id,
        signature=signature,
        model_hash=model_hash,
    )
    return to_config(config.model_dump(), prefix=MODEL_NOTARY_PREFIX)


@catch_parse_errors
def parse_model_notary_config(config: dict[str, Any]) -> ModelNotaryConfig:
    config = from_config(config)
    return ModelNotaryConfig(**config["rizemind"]["notary"]["model"])
