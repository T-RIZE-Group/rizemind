"""Flower client mod that encrypts outgoing model updates for the TEE.

Follows the same mod pattern as ``rizemind.authentication.notary.model.mod``.

On TRAIN replies this mod:

1. Extracts the TEE's public key from the fit config.
2. Loads the trainer's persistent ECDH key from their Ethereum account
   (registered on-chain, derived from mnemonic — same secp256k1 key).
3. Derives a shared secret and symmetric key with the TEE.
4. Encrypts the model parameters with AES-256-GCM.
5. Attaches ``(ciphertext, nonce, client_pubkey)`` to ``FitRes.metrics``.

Mod ordering::

    app = ClientApp(
        client_fn=client_fn,
        mods=[
            model_notary_mod,      # signs the model (runs first)
            tee_encryption_mod,    # encrypts for TEE (runs after notary)
        ],
    )
"""

import logging

from flwr.client.typing import ClientAppCallable
from flwr.common import Context, Message, log
from flwr.common.constant import MessageType
from flwr.common.recorddict_compat import (
    fitres_to_recorddict,
    recorddict_to_fitins,
    recorddict_to_fitres,
)

from rizemind.authentication.config import AccountConfig
from rizemind.tee.crypto import (
    aes_gcm_encrypt,
    derive_shared_secret,
    derive_symmetric_key,
    deserialize_public_key,
    ec_key_from_account,
    generate_ecdh_keypair,
    serialize_public_key,
)
from rizemind.tee.params import serialize_parameters
from rizemind.tee.tee_strategy import (
    TEE_CLIENT_PUBKEY_METRIC,
    TEE_ENCRYPTED_PARAMS_METRIC,
    TEE_NONCE_METRIC,
    TEE_PUBLIC_KEY_CONFIG,
)


def tee_encryption_mod(
    msg: Message,
    ctx: Context,
    call_next: ClientAppCallable,
) -> Message:
    """Flower client mod that encrypts model updates for TEE aggregation.

    The trainer's ECDH key is derived from their on-chain Ethereum account
    (via ``AccountConfig``).  This key is persistent — generated once when
    the trainer registers on the blockchain, not per round.  If no account
    config is available in the context, falls back to an ephemeral key.
    """
    reply = call_next(msg, ctx)

    if msg.metadata.message_type != MessageType.TRAIN:
        return reply

    try:
        fit_ins = recorddict_to_fitins(msg.content, True)
        tee_pubkey_bytes = fit_ins.config.get(TEE_PUBLIC_KEY_CONFIG)

        if tee_pubkey_bytes is None:
            return reply

        # Use the trainer's persistent on-chain key for ECDH.
        # The secp256k1 private key from the Ethereum account works
        # directly for ECDH — same curve, generated once at registration.
        # Falls back to an ephemeral key if account config is unavailable.
        account_config = AccountConfig.from_context(ctx)
        if account_config is not None:
            account = account_config.get_account()
            client_private = ec_key_from_account(account)
        else:
            log(
                logging.WARNING,
                "No AccountConfig in context, using ephemeral ECDH key",
            )
            client_private = generate_ecdh_keypair()

        client_public = serialize_public_key(client_private.public_key())

        # Derive shared secret with the TEE enclave
        tee_pubkey = deserialize_public_key(tee_pubkey_bytes)
        shared_secret = derive_shared_secret(client_private, tee_pubkey)
        symmetric_key = derive_symmetric_key(shared_secret)

        # Encrypt model parameters
        fit_res = recorddict_to_fitres(reply.content, False)
        param_bytes = serialize_parameters(fit_res.parameters)
        ciphertext, nonce = aes_gcm_encrypt(symmetric_key, param_bytes)

        # Attach encrypted data to metrics
        fit_res.metrics[TEE_ENCRYPTED_PARAMS_METRIC] = ciphertext
        fit_res.metrics[TEE_NONCE_METRIC] = nonce
        fit_res.metrics[TEE_CLIENT_PUBKEY_METRIC] = client_public

        reply.content = fitres_to_recorddict(fit_res, False)
        log(
            logging.INFO,
            "Encrypted model update for TEE (%d bytes ciphertext)",
            len(ciphertext),
        )

    except Exception:
        log(logging.ERROR, "Failed to encrypt model update for TEE", exc_info=True)

    return reply
