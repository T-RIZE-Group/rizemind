"""Handles the cryptographic notarization of model parameters using EIP-712 signatures.

This module provides a mechanism to ensure the authenticity and integrity of model
parameters exchanged within a Flower-based federated learning system. It integrates
EIP-712, an Ethereum standard for typed structured data signing, to allow participants
to cryptographically sign and verify model updates.
"""

from rizemind.authentication.notary.model.config import (
    ModelNotaryConfig,
    parse_model_notary_config,
    prepare_model_notary_config,
)
from rizemind.authentication.notary.model.mod import model_notary_mod
from rizemind.authentication.notary.model.model_signature import (
    hash_parameters,
    recover_model_signer,
    sign_parameters_model,
)

__all__ = [
    "ModelNotaryConfig",
    "prepare_model_notary_config",
    "parse_model_notary_config",
    "model_notary_mod",
    "hash_parameters",
    "sign_parameters_model",
    "recover_model_signer",
]
