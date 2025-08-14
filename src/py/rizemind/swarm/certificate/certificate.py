from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Final

from cryptography import x509
from cryptography.hazmat.primitives import serialization

try:
    import zstandard as _zstd
except ModuleNotFoundError:
    _zstd = None

try:
    import brotli as _brotli
except ModuleNotFoundError:
    _brotli = None

import lzma as _lzma


@unique
class Algorithm(Enum):
    """Compression algorithm identifiers with byte tags."""

    ZSTD_19 = "zstd19"
    BROTLI_11 = "brotli11"
    LZMA_9E = "lzma-9e"

    def to_byte(self) -> bytes:
        return bytes([self.to_index()])

    def to_index(self) -> int:
        """Returns a stable byte identifier (ordinal)."""
        mapping = {
            Algorithm.ZSTD_19: 0x01,
            Algorithm.BROTLI_11: 0x02,
            Algorithm.LZMA_9E: 0x03,
        }
        return mapping[self]

    @classmethod
    def from_byte(cls, b: int) -> "Algorithm":
        reverse = {
            0x01: Algorithm.ZSTD_19,
            0x02: Algorithm.BROTLI_11,
            0x03: Algorithm.LZMA_9E,
        }
        if b not in reverse:
            raise ValueError(f"Unknown algorithm byte: {b}")
        return reverse[b]


@dataclass(slots=True, frozen=True)
class CompressedCertificate:
    algorithm: Algorithm
    data: bytes

    def serialize(self) -> bytes:
        """Serializes the algorithm and data as: [1-byte algo] + data."""
        return self.algorithm.to_byte() + self.data

    @classmethod
    def deserialize(cls, raw: bytes) -> "CompressedCertificate":
        """Restores object from serialized form."""
        if len(raw) < 1:
            raise ValueError("Serialized data too short")
        algo_byte = raw[0]
        data = raw[1:]
        algorithm = Algorithm.from_byte(algo_byte)
        return cls(algorithm=algorithm, data=data)


class Certificate:
    """
    Certificate bytes with convenient
    (de)compression and persistence helpers.
    """

    __slots__ = ("_der",)

    _ZSTD_LEVEL: Final[int] = 19
    _BROTLI_QUALITY: Final[int] = 11
    _LZMA_PRESET: Final[int] = _lzma.PRESET_EXTREME | 9  # 9e

    def __init__(self, der_bytes: bytes) -> None:
        self._der: bytes = der_bytes

    @staticmethod
    def from_path(path: Path) -> "Certificate":
        """
        Load *path* (PEM **or** DER) and return a :class:`Certificate`
        containing canonical DER bytes.
        """
        raw = path.read_bytes()
        try:
            cert = x509.load_pem_x509_certificate(raw)
        except ValueError:
            try:
                cert = x509.load_der_x509_certificate(raw)
            except ValueError as exc:  # malformed file
                raise ValueError(f"Unsupported certificate format: {path}") from exc
        return Certificate(cert.public_bytes(serialization.Encoding.DER))

    def get_compressed_bytes(self) -> CompressedCertificate:
        """
        Return the certificate compressed with **the strongest algorithm
        available** on the current interpreter (zstd-19 → brotli-11 → lzma-9e).
        """
        der = self._der

        if _brotli is not None:
            return CompressedCertificate(
                Algorithm.BROTLI_11,
                _brotli.compress(der, quality=self._BROTLI_QUALITY),
            )

        if _zstd is not None:
            cctx = _zstd.ZstdCompressor(level=self._ZSTD_LEVEL)
            return CompressedCertificate(
                Algorithm.ZSTD_19,
                cctx.compress(der),
            )

        # std-lib fall-back (always available)
        return CompressedCertificate(
            Algorithm.LZMA_9E,
            _lzma.compress(der, preset=self._LZMA_PRESET),
        )

    @staticmethod
    def from_compressed(cc: CompressedCertificate) -> "Certificate":
        """Inflate *cc* and return the original :class:`Certificate`."""
        algo, data = cc.algorithm, cc.data

        if algo is Algorithm.ZSTD_19 and _zstd is not None:
            der = _zstd.ZstdDecompressor().decompress(data)
        elif algo is Algorithm.BROTLI_11 and _brotli is not None:
            der = _brotli.decompress(data)
        elif algo is Algorithm.LZMA_9E:
            der = _lzma.decompress(data)
        else:  # missing runtime dependency
            raise RuntimeError(f"Decompressor for {algo.value} not available")

        return Certificate(der)

    def store(self, path: Path) -> None:
        """
        Persist the certificate to *path* (PEM Format).
        """
        cert = x509.load_der_x509_certificate(self._der)
        pem_data = cert.public_bytes(serialization.Encoding.PEM)
        path.write_text(pem_data.decode("utf-8"))

    def __bytes__(self) -> bytes:  # convenience
        return self._der

    def __len__(self) -> int:
        return len(self._der)

    @property
    def der(self) -> bytes:
        return self._der
