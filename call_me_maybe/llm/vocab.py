""" fill me """

from call_me_maybe.parsers import input_loader


class VocabularyManager:
    """ fill me """

    def __init__(self, vocab_path: str) -> None:
        """ fill me"""

        self.vocab_path = vocab_path
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        """_summary_"""

        data = input_loader.load_json_file(self.vocab_path)
        for k, v in data:
            if isinstance(k, int) and isinstance(v, str):
                self.id_to_token[k] = v
                self.token_to_id[v] = k

    def get_token_by_id(self, id_: int) -> str | None:
        """_summary_

        Args:
            id_ (int): _description_

        Returns:
            str | None: _description_
        """

        return self.id_to_token.get(id_)

    def get_id_by_token(self, token: str) -> int | None:
        """_summary_

        Args:
            token (str): _description_

        Returns:
            int | None: _description_
        """

        return self.token_to_id.get(token)

    def vocabulary_size(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return len(self.id_to_token)