"""ECDH key exchange and AES-256-GCM encryption utilities for TEE secure aggregation.

Uses the same ``cryptography`` library already used in ``rizemind.mnemonic.store``
for AES-GCM encryption and follows the project's existing SECP256K1 curve choice
(matching Ethereum keys).
"""

import os

from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDH,
    SECP256K1,
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
    derive_private_key,
    generate_private_key,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from eth_account.signers.base import BaseAccount

# Domain separation label for HKDF, ensuring keys derived here
# cannot be confused with keys derived in other contexts.
TEE_HKDF_INFO = b"rizemind-tee-v1"

# Additional authenticated data for AES-GCM, binding ciphertext
# to the "model-params" context so it cannot be replayed elsewhere.
TEE_AES_AAD = b"model-params"


def generate_ecdh_keypair() -> EllipticCurvePrivateKey:
    """Generate a random ECDH keypair on secp256k1.

    Use this only for testing or when a persistent key is not available.
    In production, use ``ec_key_from_account`` to derive the ECDH key
    from the entity's on-chain Ethereum account.
    """
    return generate_private_key(SECP256K1())


def ec_key_from_account(account: BaseAccount) -> EllipticCurvePrivateKey:
    """Derive an ECDH-capable EC private key from an Ethereum account.

    Since Ethereum accounts use the same secp256k1 curve, the raw 32-byte
    private key can be used directly for ECDH.  This means trainers,
    evaluators, and aggregators generate their ECDH key **once** when they
    register on-chain — not per round.

    Args:
        account: An ``eth_account`` ``BaseAccount`` (e.g. from ``AccountConfig.get_account``).

    Returns:
        A ``cryptography`` EC private key suitable for ECDH on secp256k1.
    """
    private_int = int.from_bytes(account.key, "big")
    return derive_private_key(private_int, SECP256K1())


def serialize_public_key(key: EllipticCurvePublicKey) -> bytes:
    """Serialize an EC public key to uncompressed X9.62 point format."""
    return key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)


def deserialize_public_key(data: bytes) -> EllipticCurvePublicKey:
    """Deserialize an EC public key from uncompressed X9.62 point format."""
    return EllipticCurvePublicKey.from_encoded_point(SECP256K1(), data)


def derive_shared_secret(
    private_key: EllipticCurvePrivateKey,
    peer_public_key: EllipticCurvePublicKey,
) -> bytes:
    """Perform ECDH to derive a raw shared secret."""
    return private_key.exchange(ECDH(), peer_public_key)


def derive_symmetric_key(
    shared_secret: bytes,
    salt: bytes | None = None,
    info: bytes = TEE_HKDF_INFO,
    length: int = 32,
) -> bytes:
    """Derive a 256-bit AES key from an ECDH shared secret using HKDF-SHA256."""
    hkdf = HKDF(algorithm=SHA256(), length=length, salt=salt, info=info)
    return hkdf.derive(shared_secret)


def aes_gcm_encrypt(
    key: bytes, plaintext: bytes, aad: bytes = TEE_AES_AAD
) -> tuple[bytes, bytes]:
    """Encrypt with AES-256-GCM.

    Returns:
        Tuple of ``(ciphertext, nonce)``.  The nonce is 12 bytes.
    """
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return ciphertext, nonce


def aes_gcm_decrypt(
    key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes = TEE_AES_AAD
) -> bytes:
    """Decrypt with AES-256-GCM."""
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)
