"""Vocabulary loading and lookup utilities."""

from call_me_maybe.parsers import input_loader


class VocabularyManager:
    """Manage token-to-id and id-to-token vocabulary mappings."""

    def __init__(self, vocab_path: str) -> None:
        """Initialize the vocabulary manager from a JSON vocabulary file."""

        self.vocab_path = vocab_path
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        """Load vocabulary entries from the configured JSON file."""

        data = input_loader.load_json_file(self.vocab_path)
        for k, v in data:
            if isinstance(k, int) and isinstance(v, str):
                self.id_to_token[k] = v
                self.token_to_id[v] = k

    def get_token_by_id(self, id_: int) -> str | None:
        """Return the token associated with an integer ID, if present."""

        return self.id_to_token.get(id_)

    def get_id_by_token(self, token: str) -> int | None:

        """Return the integer ID associated with a token, if present."""

        return self.token_to_id.get(token)

    def vocabulary_size(self) -> int:
        """Return the number of loaded vocabulary entries."""

        return len(self.id_to_token)
