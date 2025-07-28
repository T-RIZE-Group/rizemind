from dataclasses import dataclass


@dataclass
class RizemindException(Exception):
    """
    Base exception for all errors raised by the Rizemind framework.
    """

    code: str
    message: str | None
    rizemind_error = True
