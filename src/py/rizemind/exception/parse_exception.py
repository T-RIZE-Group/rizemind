from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from pydantic_core import ValidationError as PydanticValidationError
from rizemind.exception.base_exception import RizemindException


class ParseException(RizemindException): ...


P = ParamSpec("P")
R = TypeVar("R")


def catch_parse_errors(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that wraps *func* and converts ``KeyError`` or Pydantic
    ``ValidationError`` into ``ParseException``.
    """

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (KeyError, PydanticValidationError) as exc:
            raise ParseException(code="parse_error", message=str(exc)) from exc

    return _wrapper
