#!/usr/bin/env python3
"""Shared crypto and payload helpers for the minimal Nitro demo."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KDF_INFO = b"rizemind-nitro-demo-v1"
VSOCK_PORT = 5005
AES_GCM_NONCE_SIZE = 12

LOCAL_KEYS_PATH = Path("data/nitro_demo/keys.json")
ENCLAVE_KEYS_PATH = Path("/app/data/nitro_demo/keys.json")

SENDER_PRIVATE_KEY_FIELD = "sender_private_key_hex"
SENDER_PUBLIC_KEY_FIELD = "sender_public_key_hex"
ENCLAVE_PRIVATE_KEY_FIELD = "enclave_private_key_hex"
ENCLAVE_PUBLIC_KEY_FIELD = "enclave_public_key_hex"


def default_keys_path() -> Path:
    """Return the default keys path for local runs and enclave runs."""
    env_path = os.getenv("NITRO_DEMO_KEYS_PATH")
    if env_path:
        return Path(env_path)
    if ENCLAVE_KEYS_PATH.exists():
        return ENCLAVE_KEYS_PATH
    return LOCAL_KEYS_PATH


def generate_private_key_hex() -> str:
    """Generate a secp256k1 private key as 32-byte hex."""
    private_key = ec.generate_private_key(ec.SECP256K1())
    private_value = private_key.private_numbers().private_value
    return f"{private_value:064x}"


def private_key_from_hex(private_key_hex: str) -> ec.EllipticCurvePrivateKey:
    """Construct a secp256k1 private key from hex."""
    try:
        key_int = int(private_key_hex, 16)
    except ValueError as exc:
        raise ValueError("Invalid private key hex") from exc
    try:
        return ec.derive_private_key(key_int, ec.SECP256K1())
    except ValueError as exc:
        raise ValueError("Private key is outside valid secp256k1 range") from exc


def public_key_from_hex(public_key_hex: str) -> ec.EllipticCurvePublicKey:
    """Construct a secp256k1 public key from uncompressed hex."""
    try:
        raw = bytes.fromhex(public_key_hex)
    except ValueError as exc:
        raise ValueError("Invalid public key hex") from exc
    try:
        return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), raw)
    except ValueError as exc:
        raise ValueError("Invalid secp256k1 public key") from exc


def public_key_to_hex(public_key: ec.EllipticCurvePublicKey) -> str:
    """Serialize public key to uncompressed point hex."""
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return raw.hex()


def public_key_hex_from_private_key_hex(private_key_hex: str) -> str:
    """Derive the corresponding uncompressed public key hex."""
    private_key = private_key_from_hex(private_key_hex)
    return public_key_to_hex(private_key.public_key())


def derive_shared_secret(private_key_hex: str, peer_public_key_hex: str) -> bytes:
    """Perform secp256k1 ECDH exchange."""
    private_key = private_key_from_hex(private_key_hex)
    peer_public_key = public_key_from_hex(peer_public_key_hex)
    return private_key.exchange(ec.ECDH(), peer_public_key)


def derive_symmetric_key(shared_secret: bytes) -> bytes:
    """Derive a 256-bit AES key from ECDH shared secret."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=KDF_INFO,
    )
    return hkdf.derive(shared_secret)


def _required_str_field(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid '{field}'")
    return value


def encrypt_bytes(
    plaintext: bytes,
    sender_private_key_hex: str,
    enclave_public_key_hex: str,
) -> dict[str, str]:
    """Encrypt plaintext bytes using sender->enclave ECDH."""
    shared_secret = derive_shared_secret(sender_private_key_hex, enclave_public_key_hex)
    aes_key = derive_symmetric_key(shared_secret)
    nonce = os.urandom(AES_GCM_NONCE_SIZE)
    ciphertext = AESGCM(aes_key).encrypt(nonce, plaintext, None)
    sender_public_key_hex = public_key_hex_from_private_key_hex(sender_private_key_hex)
    return {
        "sender_pubkey_hex": sender_public_key_hex,
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt_bytes(envelope: dict[str, Any], enclave_private_key_hex: str) -> bytes:
    """Decrypt encrypted envelope bytes with enclave private key."""
    sender_public_key_hex = _required_str_field(envelope, "sender_pubkey_hex")
    nonce_b64 = _required_str_field(envelope, "nonce_b64")
    ciphertext_b64 = _required_str_field(envelope, "ciphertext_b64")
    try:
        nonce = base64.b64decode(nonce_b64, validate=True)
        ciphertext = base64.b64decode(ciphertext_b64, validate=True)
    except ValueError as exc:
        raise ValueError("Invalid base64 in envelope") from exc
    if len(nonce) != AES_GCM_NONCE_SIZE:
        raise ValueError("Invalid AES-GCM nonce length")
    shared_secret = derive_shared_secret(enclave_private_key_hex, sender_public_key_hex)
    aes_key = derive_symmetric_key(shared_secret)
    return AESGCM(aes_key).decrypt(nonce, ciphertext, None)


def encrypt_number(
    number: int,
    sender_private_key_hex: str,
    enclave_public_key_hex: str,
) -> dict[str, str]:
    """Encrypt payload containing one integer field."""
    if type(number) is not int:
        raise ValueError("Input 'number' must be an integer")
    plaintext = json.dumps({"number": number}, separators=(",", ":")).encode("utf-8")
    return encrypt_bytes(plaintext, sender_private_key_hex, enclave_public_key_hex)


def decrypt_number(envelope: dict[str, Any], enclave_private_key_hex: str) -> int:
    """Decrypt envelope and extract integer payload."""
    plaintext = decrypt_bytes(envelope, enclave_private_key_hex)
    try:
        payload = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Decrypted payload is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Decrypted payload must be a JSON object")
    number = payload.get("number")
    if type(number) is not int:
        raise ValueError("Payload field 'number' must be an integer")
    return number


def _validate_key_material(data: dict[str, Any]) -> dict[str, str]:
    required_fields = [
        SENDER_PRIVATE_KEY_FIELD,
        SENDER_PUBLIC_KEY_FIELD,
        ENCLAVE_PRIVATE_KEY_FIELD,
        ENCLAVE_PUBLIC_KEY_FIELD,
    ]
    normalized: dict[str, str] = {}
    for field in required_fields:
        value = data.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing or invalid '{field}' in key material")
        normalized[field] = value
    if (
        public_key_hex_from_private_key_hex(normalized[SENDER_PRIVATE_KEY_FIELD])
        != normalized[SENDER_PUBLIC_KEY_FIELD]
    ):
        raise ValueError("Sender key pair mismatch in key material")
    if (
        public_key_hex_from_private_key_hex(normalized[ENCLAVE_PRIVATE_KEY_FIELD])
        != normalized[ENCLAVE_PUBLIC_KEY_FIELD]
    ):
        raise ValueError("Enclave key pair mismatch in key material")
    return normalized


def load_key_material(keys_path: Path | None = None) -> dict[str, str]:
    """Load and validate sender/enclave key material from disk."""
    path = keys_path or default_keys_path()
    if not path.exists():
        raise FileNotFoundError(f"Key material file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Key material file is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Key material must be a JSON object: {path}")
    return _validate_key_material(raw)


def ensure_key_material(keys_path: Path | None = None) -> dict[str, str]:
    """Create key material once, or load it when already present."""
    path = keys_path or default_keys_path()
    if path.exists():
        return load_key_material(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sender_private = generate_private_key_hex()
    enclave_private = generate_private_key_hex()
    data = {
        SENDER_PRIVATE_KEY_FIELD: sender_private,
        SENDER_PUBLIC_KEY_FIELD: public_key_hex_from_private_key_hex(sender_private),
        ENCLAVE_PRIVATE_KEY_FIELD: enclave_private,
        ENCLAVE_PUBLIC_KEY_FIELD: public_key_hex_from_private_key_hex(enclave_private),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def build_success_response(result: int) -> dict[str, Any]:
    """Build success response payload."""
    if type(result) is not int:
        raise ValueError("Result must be an integer")
    return {"ok": True, "result": result}


def build_error_response(error: str) -> dict[str, Any]:
    """Build structured error response payload."""
    return {"ok": False, "error": str(error)}


def exception_message(exc: Exception) -> str:
    """Return a readable message even for exceptions with empty ``str(exc)``."""
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__
