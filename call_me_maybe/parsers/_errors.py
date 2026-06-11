"""Internal error definitions for the parser."""

class ParserError(Exception):
    """Custom parser error to inherit from Exception"""

class ParserFileNotFoundError(ParserError):
    """ParserFileNotFoundError inherit from ParserError: if file not found"""
