"""Type identifiers used across the package.

This module exposes a small enumeration of supported primitive
types so the rest of the codebase can refer to them reliably.
"""

from enum import Enum


class Types(Enum):
    """Supported type identifiers."""

    STRING = 'string'
    INTEGER = 'integer'
