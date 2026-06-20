"""Type identifiers used across the package.

This module exposes a small enumeration of supported primitive
types so the rest of the codebase can refer to them reliably.
"""

from enum import Enum, auto


class Types(Enum):
    """Supported type identifiers."""

    STRING = 'string'
    INTEGER = 'integer'


class JSONState(Enum):
    """JSON parsing state identifiers."""

    START         = auto()  # nothing generated yet
    OBJECT_START  = auto()  # just saw {

    KEY_OPEN      = auto()  # just saw opening " of a key
    KEY_BODY      = auto()  # inside the key string characters
    KEY_CLOSE     = auto()  # just saw closing " of a key

    COLON         = auto()  # just saw :

    VALUE_STRING_OPEN  = auto()  # just saw opening " of a string value
    VALUE_STRING_BODY  = auto()  # inside the string value characters
    VALUE_STRING_CLOSE = auto()  # just saw closing " of a string value

    VALUE_NUMBER  = auto()  # generating a number value (digits)

    COMMA         = auto()  # just saw , — next key coming
    OBJECT_END    = auto()  # just saw }
    DONE          = auto()  # generation complete
