from typing import Any

from pydantic import BaseModel

from rizemind.authentication.signatures.signature import Signature
from rizemind.configuration.transform import from_config, to_config
from rizemind.contracts.erc.erc5267.typings import EIP712DomainMinimal
from rizemind.exception.parse_exception import catch_parse_errors

MODEL_NOTARY_PREFIX = "rizemind.notary.model"


class ModelNotaryConfig(BaseModel):
    """Configuration for cryptographically attesting a model's parameters verification.

    This configuration bundles a model's cryptographic hash with its digital signature and the
    `EIP712 Domain`, enabling verifiable notarization for a specific round.

    Attributes:
        domain: EIP712 Domain
        round_id: The specific federated learning round number this notarization applies to.
        model_hash: The cryptographic hash of the model's parameters.
        signature: The digital signature attesting to the model's hash.
    """

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
    """Constructs and serializes a `ModelNotaryConfig` into a nested dictionary.

    This function takes the core components of a model notarization, creates a `ModelNotaryConfig`
    object, and then transforms it into a standardized dictionary format prefixed with
    `MODEL_NOTARY_PREFIX`, to comply with flower's `Config`.

    Args:
        round_id: The current federated learning round number.
        domain: The EIP-712 domain.
        signature: The digital signature of the model hash.
        model_hash: The hash of the model's parameters.

    Returns:
        Config: The serialized `ModelNotaryConfig` data.
    """
    config = ModelNotaryConfig(
        domain=domain,
        round_id=round_id,
        signature=signature,
        model_hash=model_hash,
    )
    return to_config(config.model_dump(), prefix=MODEL_NOTARY_PREFIX)


@catch_parse_errors
def parse_model_notary_config(config: dict[str, Any]) -> ModelNotaryConfig:
    """Parses a configuration dictionary into a `ModelNotaryConfig` object.

    This function acts as the inverse of `prepare_model_notary_config`. It takes a standardized
    configuration dictionary, extracts the relevant `MODEL_NOTARY_PREFIX` section, and
    deserializes it into a structured `ModelNotaryConfig`.

    Args:
        config: The input configuration dictionary, expected to contain the
        notarization data under the 'MODEL_NOTARY_PREFIX' key.

    Returns:
        ModelNotaryConfig: A`ModelNotaryConfig` populated with the data from
        the input dictionary.

    Raises:
        ParseException: If the notary configuration cannot be parsed.
    """
    config = from_config(config)
    return ModelNotaryConfig(**config["rizemind"]["notary"]["model"])
