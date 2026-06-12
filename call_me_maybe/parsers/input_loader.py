"""Input loader for JSON files used by the parsers package."""

import json

from call_me_maybe.parsers import _errors


def load_json_file(name: str) -> list:
    """Load a JSON file and return its parsed content.

    Args:
        name: Path to the JSON file to load.

    Raises:
        _errors.ParserFileNotFoundError: If the file does not exist.
        _errors.ParserPermissionError: If the file cannot be opened due to
            permission issues.
        _errors.ParserJSONDecodeError: If the file contents are not valid
            JSON.

    Returns:
        The parsed JSON content (typically a list or dict depending on the
        file contents).
    """

    try:
        with open(name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data

    except FileNotFoundError as exc:
        raise _errors.ParserFileNotFoundError(
            f"File not found: {name}") from exc
    except PermissionError as exc:
        raise _errors.ParserPermissionError(
            f"Permission denied: {name}") from exc
    except json.JSONDecodeError as exc:
        raise _errors.ParserJSONDecodeError(
            f"Invalid JSON in file: {name}") from exc
