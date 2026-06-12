"""Define the FunctionDefinition blueprint object."""

from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict, field_validator

from call_me_maybe._types._types import Types
from call_me_maybe.parsers import _errors


class Parameter(BaseModel):
    """Parameter in FunctionDefinition definition schema."""

    model_config = ConfigDict(extra='forbid')

    name: Annotated[str, Field(min_length=1, max_length=20)]
    type_: Types

    @field_validator('type_', mode='before')
    @staticmethod
    def validate_type(type_: object) -> Types:
        """Validate and convert a parameter type string to a Types enum.

        Args:
            type_ (str): The parameter type as a string 'string' or 'number'

        Raises:
            _errors.ParserValidationError: When invalid parameter type.

        Returns:
            Types: The corresponding Types enum value.
        """

        if isinstance(type_, str) and type_.lower() == 'string':
            return Types.STRING
        if isinstance(type_, str) and type_.lower() == 'number':
            return Types.INTEGER
        raise _errors.ParserValidationError(f'Invalid parameter type {type_}')

    def __repr__(self) -> str:
        return f"-> {self.name}"


class FunctionDefinition(BaseModel):
    """Function definition schema."""

    model_config = ConfigDict(extra='forbid')

    name: Annotated[str, Field(min_length=3, max_length=40)]
    description: Annotated[str, Field(min_length=3, max_length=200)]
    parameters: list[Parameter]
    return_: Types

    @field_validator('return_', mode='before')
    @staticmethod
    def validate_return(return_: object) -> Types:
        """Validate and convert a return type string to a Types enum.

        Args:
            return_ (str): The return type as a string (e.g. 'string',
                'number').

        Raises:
            _errors.ParserValidationError: If the provided type string
                is not supported.

        Returns:
            Types: The corresponding Types enum value.
        """

        if isinstance(return_, str) and return_.lower() == 'string':
            return Types.STRING
        if isinstance(return_, str) and return_.lower() == 'number':
            return Types.INTEGER
        raise _errors.ParserValidationError(f'Invalid return type {return_}')

    def __repr__(self) -> str:
        return f"-> {self.name}"
