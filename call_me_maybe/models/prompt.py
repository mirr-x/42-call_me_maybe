"""Define Prompt blueprint object."""

class Prompt:
    """Represents a prompt object for storing and managing prompt text."""

    def __init__(self, prompt: str) -> None:
        """Initialize a Prompt instance.

        Args:
            prompt (str): The prompt text to store.
        """

        self.prompt = prompt

    def __repr__(self) -> str:
        return f'-> {self.prompt}'
