"""Internal error definitions for the parser."""


class ParserError(Exception):
    """Custom parser error to inherit from Exception"""


class ParserFileNotFoundError(ParserError):
    """ParserFileNotFoundError inherit from ParserError: if file not found"""


class ParserPermissionError(ParserError):
    """ParserPermissionError inherit from ParserError: if user has no
    perrmision"""


class ParserJSONDecodeError(ParserError):
    """ParserJSONDecodeError inherit from ParserError: if the file is not in
    json format"""


class ParserInvalidformat(ParserError):
    """ParserInvalidformat inherit from ParserError: if format invalid
    inside the file"""


class ParserInvalidValue(ParserError):
    """ParserInvalidValue inherit from ParserError: if value is invalid
    inside the file"""


class ParserValidationError(ParserError):
    """ParserValidationError inherit from ParserError: if validation failed
    in pydantic"""
