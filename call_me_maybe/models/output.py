"""Output model for a prompt and function metadata."""

from typing import Any

from call_me_maybe.models.prompt import Prompt


class OutputModel:
    """Represents the final output structure."""

    def __init__(self, prompt: Prompt, name: str, parameters: dict[str, Any]) -> None:
        """Initialize the output model.

        Args:
            prompt (Prompt): The prompt instance.
            name (str): The function name.
            parameters (dict[str, Any]): The function parameters.
        """

        self.prompt = prompt
        self.name = name
        self.parameters = parameters
