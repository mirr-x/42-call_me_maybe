"""Validator utilities for parsing function definitions."""

from typing import Any

from call_me_maybe.models.function import FunctionDefinition, Parameter
from call_me_maybe.models.prompt import Prompt
from call_me_maybe.parsers._errors import ParserInvalidformat


def _extract_parameters(parameters: dict[str, Any]) -> list[Parameter]:
    """Convert a mapping of parameter names to types into Parameter objects.

    Args:
        parameters: A mapping where keys are parameter names and values are
            their type representation (string or other serializable form).

    Returns:
        A list of Parameter instances constructed from the mapping.
    """

    valid_parameters: list[Parameter] = []
    for name, definition in parameters.items():
        valid_parameters.append(Parameter(name=name, type_=definition['type']))
    return valid_parameters


def validate_functions(
        functions_list: list[dict[str, Any]]
        ) -> list[FunctionDefinition]:
    """Validate and convert a list of raw function dicts to FunctionDefinition.

    Args:
        functions_list: List of dictionaries describing functions.

    Returns:
        list[FunctionDefinition]: A list of FunctionDefinition instances.
    """

    try:
        valid_functions: list[FunctionDefinition] = []
        for function in functions_list:
            name = function['name']
            description = function['description']
            parameters = _extract_parameters(function['parameters'])
            return_ = (function['returns'])['type']

            valid_functions.append(
                FunctionDefinition(
                    name=name,
                    description=description,
                    parameters=parameters,
                    return_=return_
                    )
                )
        return valid_functions
    except KeyError as cause:
        raise ParserInvalidformat('Invalid Format Function') from cause


def validate_prompts(
        prompts_list: list[dict[str, Any]]
        ) -> list[Prompt]:
    """Validate and convert a list of raw prompts dicts to Prompt.

    Args:
        prompts_list: List of dictionaries describing prompts.

    Returns:
        list[Prompts]: A list of Prompts instances.
    """

    try:
        valid_prompts: list[Prompt] = []
        for prompt in prompts_list:
            prompt = prompt['prompt']
            valid_prompts.append(Prompt(prompt=prompt))

        return valid_prompts
    except KeyError as cause:
        raise ParserInvalidformat('Invalid Format Prompt') from cause
