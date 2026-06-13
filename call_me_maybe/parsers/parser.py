"""Parser for loading and validating functions and prompts from JSON files."""

import pydantic


from call_me_maybe.models.function import FunctionDefinition
from call_me_maybe.models.prompt import Prompt
from call_me_maybe.parsers.input_loader import load_json_file
from call_me_maybe.parsers import _errors
from call_me_maybe.parsers.validator import (
    validate_functions,
    validate_prompts
)


class Parsing:
    """Load and validate function and prompt definitions from files."""

    def __init__(
            self, file_name_functions: str, file_name_prompt: str) -> None:
        """Initialize the parser with file paths.

        Args:
            file_name_functions: path to functions JSON file.
            file_name_prompt: path to prompts JSON file.
        """

        self.file_name_functions = file_name_functions
        self.file_name_prompt = file_name_prompt
        self._functions: list[FunctionDefinition]
        self._prompts: list[Prompt]

    def run(self):
        """Load and validate the JSON files and store results.

        Returns:
            A tuple (functions, prompts) of validated objects.

        Raises:
            _errors.ParserValidationError: if validation fails.
        """

        try:
            # get all raw function and prompt (json format) from file
            functions_list = load_json_file(self.file_name_functions)
            prompt_list = load_json_file(self.file_name_prompt)

            # parse them all into objects and store them
            self.functions = validate_functions(functions_list)
            self.prompts = validate_prompts(prompt_list)

        except pydantic.ValidationError as cause:
            raise _errors.ParserValidationError('Validation Error') from cause

    def get_function(self) -> list[FunctionDefinition]:
        """Get the validated function definitions.

        Returns:
            list[FunctionDefinition]: The validated function definitions.
        """
        return self.functions

    def get_prompts(self) -> list[Prompt]:
        """Get the validated prompt definitions.

        Returns:
            list[Prompt]: The validated prompt definitions.
        """
        return self.prompts
