from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from pydantic_core import ValidationError as PydanticValidationError
from rizemind.exception.base_exception import RizemindException


class ParseException(RizemindException): ...


_F = TypeVar("_F", bound=Callable[..., object])


def catch_parse_errors(func: _F) -> _F:  # type: ignore[misc]
    """
    Decorator that wraps *func* and converts ``KeyError`` or Pydantic
    ``ValidationError`` into ``ParseException``.
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyError, PydanticValidationError) as exc:
            raise ParseException(code="parse_error", message=str(exc)) from exc

    # mypy / typing friendly cast
    return _wrapper  # type: ignore[return-value]
