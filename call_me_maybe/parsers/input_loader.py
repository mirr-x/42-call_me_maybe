""" fill me """

from call_me_maybe.parsers import _errors


def load_file(name: str) -> None:
    """ fill me """
    try:
        with open(name, 'r', encoding='utf-8') as f:
            file = f
    except FileNotFoundError as exc:
        raise _errors.ParserFileNotFoundError(f'File not found {name}') from exc

