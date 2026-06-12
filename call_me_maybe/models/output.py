"""Output model for a prompt and function metadata."""

from typing import Annotated, Any
from pydantic import BaseModel, Field, ConfigDict

from call_me_maybe.models.prompt import Prompt


class OutputModel(BaseModel):
    """Represents the final output structure."""

    model_config = ConfigDict(extra='forbid')

    prompt: Prompt
    name: Annotated[str, Field(min_length=3, max_length=10)]
    parameters: dict[str, Any]

    def __repr__(self) -> str:
        return f"-> {self.prompt}"
