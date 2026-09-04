"""Redaction of secret values before configuration leaves the process.

Configuration reaches Rizemind in two shapes: nested mappings, such as a parsed
``pyproject.toml``, and flat mappings with dot-delimited keys, such as a Flower
``run_config``. Either can carry credentials — a BIP-39 mnemonic, a keystore
passphrase — so anything that persists or transmits configuration should pass it
through `redact` first.

Redaction matches on key names rather than on values, and the names in
`SECRET_FIELD_NAMES` are the contract: a credential stored under a name that is
not listed will not be caught. This is a safety net at the boundary, not a
substitute for keeping credentials out of configuration in the first place.

Typical usage example:

    >>> redact({"eth": {"account": {"mnemonic": "test test junk"}}})
    {'eth': {'account': {'mnemonic': '***REDACTED***'}}}
    >>> redact({"eth.account.mnemonic": "test test junk", "web3.url": "http://rpc"})
    {'eth.account.mnemonic': '***REDACTED***', 'web3.url': 'http://rpc'}
"""

from collections.abc import Mapping
from typing import Any

REDACTED = "***REDACTED***"
"""The value substituted for every secret."""

SECRET_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "mnemonic",
        "passphrase",
        "password",
        "private_key",
        "secret",
        "seed_phrase",
        "token",
    }
)
"""Key names whose values are treated as secret.

A key matches when its last dot-delimited segment equals one of these names, or
ends with one of them preceded by an underscore — so ``rizenet_mnemonic`` matches
but ``mnemonic_store`` does not, the latter being a subtree whose own
``passphrase`` key is matched instead.
"""


def _normalize(segment: str) -> str:
    """Fold a key segment to the form used for matching."""
    return segment.strip().casefold().replace("-", "_")


def is_secret_key(key: str) -> bool:
    """Report whether a key names a secret.

    Only the last dot-delimited segment is considered, so a flat key such as
    ``eth.account.mnemonic`` matches on ``mnemonic``. Matching ignores case and
    treats hyphens and underscores alike.

    Args:
        key: A configuration key, either a bare name or a dot-delimited path.

    Returns:
        True if the key's value should be redacted.
    """
    segment = _normalize(key.rsplit(".", 1)[-1])
    return any(
        segment == name or segment.endswith(f"_{name}") for name in SECRET_FIELD_NAMES
    )


def redact(config: Mapping[str, Any], *, placeholder: str = REDACTED) -> dict[str, Any]:
    """Copy a configuration mapping with every secret value replaced.

    Nested mappings and lists are walked, so both nested and flat dot-delimited
    layouts are covered. A secret key whose value is itself a mapping has the
    whole subtree replaced. The input is not modified.

    Args:
        config: The configuration to copy.
        placeholder: The value substituted for each secret.

    Returns:
        A new mapping with the same structure and secrets replaced.
    """
    return {
        key: placeholder if is_secret_key(key) else _redact_value(value, placeholder)
        for key, value in config.items()
    }


def _redact_value(value: Any, placeholder: str) -> Any:
    """Redact secrets nested inside a single configuration value."""
    if isinstance(value, Mapping):
        return redact(value, placeholder=placeholder)
    if isinstance(value, list):
        return [_redact_value(item, placeholder) for item in value]
    return value
