"""Type identifiers used across the package.

This module exposes a small enumeration of supported primitive
types so the rest of the codebase can refer to them reliably.
"""

from enum import Enum, auto


class Types(Enum):
    """Supported type identifiers."""

    STRING = 'string'
    INTEGER = 'integer'


class JsonState(Enum):
    """JSON parsing state identifiers."""

    START = auto()
    WAITING_FOR_KEY = auto()
    WAITING_FOR_COLON = auto()
    WAITING_FOR_VALUE = auto()
    WAITING_FOR_NEXT = auto()
    DONE = auto()
