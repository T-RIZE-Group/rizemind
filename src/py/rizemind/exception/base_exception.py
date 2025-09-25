from dataclasses import dataclass


@dataclass
class RizemindException(Exception):
    """The common base for all Rizemind framework errors.

    Attributes:
        code: Error code.
        message: Optional human-readable description.
    """

    code: str
    message: str | None
