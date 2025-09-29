import base64
import json
import os
from pathlib import Path
from unicodedata import normalize

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from eth_account.hdaccount import generate_mnemonic

from rizemind.constants import RIZEMIND_HOME


class MnemonicStore:
    """Manages the secure storage of BIP39 mnemonic phrases."""

    _keystore_dir: Path

    def __init__(self, keystore_dir=RIZEMIND_HOME / "keystore") -> None:
        """Initializes the MnemonicStore.

        Args:
            keystore_dir: The directory to store mnemonic keystore files.
        """
        self._keystore_dir = keystore_dir
        keystore_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, words=24) -> str:
        """Generates a new BIP39 mnemonic phrase.

        Args:
            words: The number of words in the mnemonic (e.g., 12, 15, 18, 21, 24).

        Returns:
            A new mnemonic phrase.
        """
        return generate_mnemonic(lang="english", num_words=words)

    def save(self, account_name: str, passphrase: str, mnemonic: str) -> Path:
        """Encrypts and saves a mnemonic to a keystore file.

        Args:
            account_name: The name of the account to save.
            passphrase: The passphrase to encrypt the mnemonic.
            mnemonic: The mnemonic phrase to save.

        Returns:
            The path to the newly created keystore file.
        """
        encrypted = self._encrypt_mnemonic(mnemonic, passphrase)
        file_path = self.get_keystore_file(account_name)
        file_path.write_text(json.dumps(encrypted))
        return file_path

    def get_keystore_dir(self) -> Path:
        """Returns the keystore directory path."""
        return self._keystore_dir

    def get_keystore_file(self, account_name: str) -> Path:
        """Returns the path to an account's keystore file.

        Args:
            account_name: The name of the account.

        Returns:
            The keystore file path for the given account.
        """
        keystore_dir = self.get_keystore_dir()
        return keystore_dir / f"{account_name}.json"

    def exists(self, account_name: str) -> bool:
        """Checks if a keystore for an account name exists.

        Args:
            account_name: The name of the account.

        Returns:
            True if the keystore file exists, False otherwise.
        """
        keystore = self.get_keystore_file(account_name)
        return keystore.exists()

    def load(self, account_name: str, passphrase: str) -> str:
        """Loads and decrypts a mnemonic from a keystore file.

        Args:
            account_name: The name of the account to load.
            passphrase: The passphrase to decrypt the mnemonic.

        Returns:
            The decrypted mnemonic phrase.

        Raises:
            FileNotFoundError: If the keystore for the account does not exist.
            ValueError: If decryption fails due to an incorrect passphrase or
                corrupted data.
        """
        if not self.exists(account_name):
            raise FileNotFoundError(f"'{account_name}' does not exist")

        data = json.loads(self.get_keystore_file(account_name).read_text())
        return self._decrypt_mnemonic(data, passphrase)

    def list_accounts(self) -> list[str]:
        """Lists all available account names from the keystore directory.

        Returns:
            A sorted list of account names.
        """
        keystore_dir = self.get_keystore_dir()
        return sorted(
            p.stem  # file name minus “.json”
            for p in keystore_dir.glob("*.json")  # only keystore files
            if p.is_file()
        )

    @staticmethod
    def _derive_key(passphrase: str, salt: bytes, length: int = 32) -> bytes:
        """Derives a key from a passphrase using Scrypt.

        Args:
            passphrase: The passphrase to derive the key from.
            salt: The salt to use for key derivation.
            length: The desired length of the derived key in bytes.

        Returns:
            The derived key.
        """
        kdf = Scrypt(salt=salt, length=length, n=2**15, r=8, p=1)
        return kdf.derive(passphrase.encode())

    def _encrypt_mnemonic(self, mnemonic: str, passphrase: str) -> dict:
        """Encrypts a mnemonic using AES-GCM.

        Args:
            mnemonic: The mnemonic phrase to encrypt.
            passphrase: The passphrase to use for encryption.

        Returns:
            A dictionary containing the encrypted data and parameters.
        """
        salt = os.urandom(16)
        nonce = os.urandom(12)
        key = self._derive_key(normalize("NFKC", passphrase), salt)

        aesgcm = AESGCM(key)
        cipher = aesgcm.encrypt(nonce, mnemonic.encode("utf-8"), b"mnemonic")

        return {
            "version": 1,
            "kdf": "scrypt",
            "n": 1 << 15,
            "r": 8,
            "p": 1,
            "salt": base64.b64encode(salt).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "cipher": base64.b64encode(cipher).decode(),
            "cipher_algo": "AES-256-GCM",
            "aad": "mnemonic",
        }

    def _decrypt_mnemonic(self, blob: dict, passphrase: str) -> str:
        """Decrypts a mnemonic from an encrypted blob.

        Args:
            blob: The dictionary containing the encrypted data.
            passphrase: The passphrase to use for decryption.

        Returns:
            The decrypted mnemonic phrase.

        Raises:
            ValueError: If decryption fails due to an incorrect passphrase or
                corrupted data.
        """
        try:
            salt = base64.b64decode(blob["salt"])
            nonce = base64.b64decode(blob["nonce"])
            cipher = base64.b64decode(blob["cipher"])

            key = self._derive_key(normalize("NFKC", passphrase), salt)

            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, cipher, b"mnemonic")
            return plaintext.decode("utf-8")

        except InvalidTag:
            raise ValueError(
                "Decryption failed: incorrect pass-phrase or corrupted data"
            )
