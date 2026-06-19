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

    START_OBJECT = 'start_object'
    START = 'start'

    NAME = 'name'
    STRING = 'string'
    END_KEY = 'end_key'
    NUMBER = 'number'
    COLON = 'colon'
    END = 'end'
