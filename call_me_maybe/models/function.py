""" Define FunctionDefinition blueprint Object. """

from typing import Any

class FunctionDefinition:
    """Blueprint for defining a callable function."""

    def __init__(
            self,
            name: str,
            description: str,
            parameters: dict[str, Any],
            return_: dict[str, Any]
            ) -> None:
        """Initialize a FunctionDefinition.

        Args:
            name (str): The name of the function
            description (str): A description of what the function does
            parameters (dict[str, Any]): The parameters the function accepts
            return_ (dict[str, Any]): The return value description and type
        """

        self.name = name
        self.description = description
        self.parameters = parameters
        self.return_ = return_

    def __repr__(self) -> str:
        return f'-> {self.name}'
