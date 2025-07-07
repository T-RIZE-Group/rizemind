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
    _keystore_dir: Path

    def __init__(self, keystore_dir=RIZEMIND_HOME / "keystore") -> None:
        self._keystore_dir = keystore_dir
        keystore_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, words=24) -> str:
        return generate_mnemonic(lang="english", num_words=words)

    def save(self, account_name: str, passphrase: str, mnemonic: str) -> Path:
        """
        Encrypts the *mnemonic* with the given pass-phrase and write a
        keystore JSON.  Returns the file path.
        """
        encrypted = self._encrypt_mnemonic(mnemonic, passphrase)
        file_path = self.get_keystore_file(account_name)
        file_path.write_text(json.dumps(encrypted))
        return file_path

    def get_keystore_dir(self) -> Path:
        return self._keystore_dir

    def get_keystore_file(self, account_name: str) -> Path:
        keystore_dir = self.get_keystore_dir()
        return keystore_dir / f"{account_name}.json"

    def exists(self, account_name: str) -> bool:
        keystore = self.get_keystore_file(account_name)
        return keystore.exists()

    def load(self, account_name: str, passphrase: str) -> str:
        """
        Unlock the keystore, recover the mnemonic, and rebuild the account.
        """
        if not self.exists(account_name):
            raise FileNotFoundError(f"'{account_name}' does not exist")

        data = json.loads(self.get_keystore_file(account_name).read_text())
        return self._decrypt_mnemonic(data, passphrase)

    def list_accounts(self) -> list[str]:
        """
        Return all account names that have a keystore JSON
        in ``<home>/.rzmnd/keystore/``.

        Example
        -------
        >>> am = AccountManager()
        >>> am.list_accounts()
        ['alice', 'bob', 'test-net']
        """
        keystore_dir = self.get_keystore_dir()
        return sorted(
            p.stem  # file name minus “.json”
            for p in keystore_dir.glob("*.json")  # only keystore files
            if p.is_file()
        )

    @staticmethod
    def _derive_key(passphrase: str, salt: bytes, length: int = 32) -> bytes:
        kdf = Scrypt(salt=salt, length=length, n=2**15, r=8, p=1)
        return kdf.derive(passphrase.encode())

    def _encrypt_mnemonic(self, mnemonic: str, passphrase: str) -> dict:
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
