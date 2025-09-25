"""Exception classes for the Rizemind framework.

This module provides custom exception types used throughout the Rizemind
framework for error handling and reporting.
"""

from rizemind.exception.base_exception import RizemindException
from rizemind.exception.parse_exception import ParseException

__all__ = ["RizemindException", "ParseException"]
