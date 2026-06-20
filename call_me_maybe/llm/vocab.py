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

    def build_mappings(self, data: dict) -> None:
        """Build token-to-id and id-to-token mappings from loaded data."""

        for k, v in data.items():
            token: str | None = None
            token_id: int | None = None

            if isinstance(k, str) and isinstance(v, int):
                token = k
                token_id = v
            elif isinstance(k, str) and k.isdigit() and isinstance(v, str):
                token = v
                token_id = int(k)

            if token is not None and token_id is not None:
                self.id_to_token[token_id] = token
                self.token_to_id[token] = token_id

    def _load_vocabulary(self) -> None:
        """Load vocabulary entries from the configured JSON file."""

        data = input_loader.load_json_file(self.vocab_path)
        self.build_mappings(data)

    def get_token_by_id(self, id_: int) -> str:
        """Return the token associated with an integer ID, if present."""

        return self.id_to_token.get(id_, "<UNK>")

    @staticmethod
    def _display_token(token: str) -> str:
        """Convert tokenizer-specific markers into a more readable token."""

        leading_space_markers = len(token) - len(token.lstrip("Ġ"))
        if leading_space_markers:
            return (" " * leading_space_markers) + token[leading_space_markers:]
        return token

    def get_display_token_by_id(self, id_: int) -> str:
        """Return a human-readable token for display/debugging."""

        return self._display_token(self.get_token_by_id(id_))

    def get_id_by_token(self, token: str) -> int:

        """Return the integer ID associated with a token, if present."""

        return self.token_to_id.get(token, -1)

    def inspect_token(self, token_id: int) -> str:
        """Format a single vocabulary entry for debugging."""

        token = self.get_token_by_id(token_id)
        return (
            f"TOKEN ID: {token_id}\n"
            f"TOKEN: {self.get_display_token_by_id(token_id)}\n"
            f"RAW TOKEN: {token}"
        )

    def inspect_tokens(self, token_ids: list[int]) -> list[str]:
        """Format multiple vocabulary entries for debugging."""

        return [self.inspect_token(token_id) for token_id in token_ids]

    def vocabulary_size(self) -> int:
        """Return the number of loaded vocabulary entries."""

        return len(self.id_to_token)
