from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from pydantic_core import ValidationError as PydanticValidationError

from rizemind.exception.base_exception import RizemindException


class ParseException(RizemindException):
    """A Pydantic model parse error."""

    ...


P = ParamSpec("P")
R = TypeVar("R")


def catch_parse_errors(func: Callable[P, R]) -> Callable[P, R]:
    """Wrap a callable and convert common parse errors to `ParseException`.

    Args:
        func: The function to wrap.

    Returns:
        A callable that behaves like `func` but raises `ParseException` when
        a `KeyError` or Pydantic `ValidationError` occurs.

    Raises:
        ParseException: If a `KeyError` or Pydantic `ValidationError` is raised
            by the wrapped callable.
    """

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (KeyError, PydanticValidationError) as exc:
            raise ParseException(code="parse_error", message=str(exc)) from exc

    return _wrapper
