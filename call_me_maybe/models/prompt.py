"""Define Prompt blueprint object."""

from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict


class Prompt(BaseModel):
    """Represents a prompt object for storing and managing prompt text."""

    model_config = ConfigDict(extra='forbid')

    prompt: Annotated[str, Field(min_length=3, max_length=100)]

    def __repr__(self) -> str:
        return f'-> {self.prompt}'
