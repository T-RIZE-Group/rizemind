"""Tests for ECDH key exchange and AES-GCM encryption utilities."""

import pytest

from eth_account import Account

from rizemind.tee.crypto import (
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    derive_shared_secret,
    derive_symmetric_key,
    deserialize_public_key,
    ec_key_from_account,
    generate_ecdh_keypair,
    serialize_public_key,
)


class TestECDHKeyExchange:
    def test_keypair_generation(self):
        key = generate_ecdh_keypair()
        assert key is not None
        pub_bytes = serialize_public_key(key.public_key())
        # Uncompressed point: 1 byte prefix (0x04) + 32 bytes x + 32 bytes y
        assert len(pub_bytes) == 65
        assert pub_bytes[0] == 0x04

    def test_public_key_round_trip(self):
        key = generate_ecdh_keypair()
        pub_bytes = serialize_public_key(key.public_key())
        recovered = deserialize_public_key(pub_bytes)
        assert serialize_public_key(recovered) == pub_bytes

    def test_shared_secret_agreement(self):
        """Both parties derive the same shared secret."""
        alice = generate_ecdh_keypair()
        bob = generate_ecdh_keypair()

        secret_alice = derive_shared_secret(alice, bob.public_key())
        secret_bob = derive_shared_secret(bob, alice.public_key())

        assert secret_alice == secret_bob
        assert len(secret_alice) == 32  # secp256k1 produces 32-byte secrets

    def test_different_keypairs_different_secrets(self):
        alice = generate_ecdh_keypair()
        bob = generate_ecdh_keypair()
        charlie = generate_ecdh_keypair()

        secret_ab = derive_shared_secret(alice, bob.public_key())
        secret_ac = derive_shared_secret(alice, charlie.public_key())

        assert secret_ab != secret_ac


class TestSymmetricKeyDerivation:
    def test_deterministic(self):
        shared_secret = b"\x42" * 32
        key1 = derive_symmetric_key(shared_secret)
        key2 = derive_symmetric_key(shared_secret)
        assert key1 == key2

    def test_key_length(self):
        key = derive_symmetric_key(b"\x42" * 32)
        assert len(key) == 32  # AES-256

    def test_different_info_different_key(self):
        shared = b"\x42" * 32
        key1 = derive_symmetric_key(shared, info=b"context-a")
        key2 = derive_symmetric_key(shared, info=b"context-b")
        assert key1 != key2


class TestAESGCM:
    def test_encrypt_decrypt_round_trip(self):
        key = derive_symmetric_key(b"\x42" * 32)
        plaintext = b"hello world model parameters"

        ciphertext, nonce = aes_gcm_encrypt(key, plaintext)
        recovered = aes_gcm_decrypt(key, nonce, ciphertext)

        assert recovered == plaintext

    def test_ciphertext_differs_from_plaintext(self):
        key = derive_symmetric_key(b"\x42" * 32)
        plaintext = b"model weights here"
        ciphertext, _ = aes_gcm_encrypt(key, plaintext)
        assert ciphertext != plaintext

    def test_wrong_key_fails(self):
        key1 = derive_symmetric_key(b"\x42" * 32)
        key2 = derive_symmetric_key(b"\x43" * 32)
        plaintext = b"secret data"

        ciphertext, nonce = aes_gcm_encrypt(key1, plaintext)

        with pytest.raises(Exception):
            aes_gcm_decrypt(key2, nonce, ciphertext)

    def test_tampered_ciphertext_fails(self):
        key = derive_symmetric_key(b"\x42" * 32)
        plaintext = b"model data"

        ciphertext, nonce = aes_gcm_encrypt(key, plaintext)
        # Tamper with ciphertext
        tampered = bytearray(ciphertext)
        tampered[0] ^= 0xFF
        tampered = bytes(tampered)

        with pytest.raises(Exception):
            aes_gcm_decrypt(key, nonce, tampered)

    def test_large_payload(self):
        """Simulates encrypting large model parameters."""
        key = derive_symmetric_key(b"\x42" * 32)
        plaintext = b"\x00" * (1024 * 1024)  # 1 MB

        ciphertext, nonce = aes_gcm_encrypt(key, plaintext)
        recovered = aes_gcm_decrypt(key, nonce, ciphertext)

        assert recovered == plaintext


class TestEndToEndECDHEncryption:
    def test_full_flow(self):
        """Trainer encrypts data for TEE, TEE decrypts — full ECDH flow."""
        # TEE generates keypair
        tee_key = generate_ecdh_keypair()
        tee_pubkey_bytes = serialize_public_key(tee_key.public_key())

        # Trainer generates ephemeral keypair
        trainer_key = generate_ecdh_keypair()
        trainer_pubkey_bytes = serialize_public_key(trainer_key.public_key())

        # Trainer derives shared secret + symmetric key
        tee_pubkey = deserialize_public_key(tee_pubkey_bytes)
        trainer_shared = derive_shared_secret(trainer_key, tee_pubkey)
        trainer_sym_key = derive_symmetric_key(trainer_shared)

        # Trainer encrypts
        plaintext = b"model weights: [1.0, 2.0, 3.0]"
        ciphertext, nonce = aes_gcm_encrypt(trainer_sym_key, plaintext)

        # TEE derives the same shared secret + symmetric key
        trainer_pubkey = deserialize_public_key(trainer_pubkey_bytes)
        tee_shared = derive_shared_secret(tee_key, trainer_pubkey)
        tee_sym_key = derive_symmetric_key(tee_shared)

        # TEE decrypts
        recovered = aes_gcm_decrypt(tee_sym_key, nonce, ciphertext)

        assert recovered == plaintext


class TestEcKeyFromAccount:
    def test_deterministic_key_from_mnemonic(self):
        """Same mnemonic always produces the same ECDH key."""
        mnemonic = "test test test test test test test test test test test junk"
        Account.enable_unaudited_hdwallet_features()
        acct1 = Account.from_mnemonic(mnemonic)
        acct2 = Account.from_mnemonic(mnemonic)

        key1 = ec_key_from_account(acct1)
        key2 = ec_key_from_account(acct2)

        assert serialize_public_key(key1.public_key()) == serialize_public_key(
            key2.public_key()
        )

    def test_different_accounts_different_keys(self):
        """Different HD wallet indices produce different ECDH keys."""
        mnemonic = "test test test test test test test test test test test junk"
        Account.enable_unaudited_hdwallet_features()
        acct0 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/0'/0/0")
        acct1 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/0'/0/1")

        key0 = ec_key_from_account(acct0)
        key1 = ec_key_from_account(acct1)

        assert serialize_public_key(key0.public_key()) != serialize_public_key(
            key1.public_key()
        )

    def test_account_key_works_for_ecdh(self):
        """Account-derived key can perform ECDH with a TEE key."""
        mnemonic = "test test test test test test test test test test test junk"
        Account.enable_unaudited_hdwallet_features()
        acct = Account.from_mnemonic(mnemonic)

        trainer_key = ec_key_from_account(acct)
        tee_key = generate_ecdh_keypair()

        # Both sides derive the same shared secret
        secret_trainer = derive_shared_secret(trainer_key, tee_key.public_key())
        secret_tee = derive_shared_secret(tee_key, trainer_key.public_key())

        assert secret_trainer == secret_tee
