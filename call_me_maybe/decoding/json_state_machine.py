"""State machine for validating simplified JSON token transitions."""


from typing import TYPE_CHECKING

from call_me_maybe._types._types import JSONState
from call_me_maybe.parsers import _errors

if TYPE_CHECKING:
    from call_me_maybe.decoding.constraints import ConstraintEngine



class JSONStateMachine:
    """Tracks parsing state while consuming JSON-like tokens."""

    def __init__(self, constrained_engein: "ConstraintEngine") -> None:
        self.state = JSONState.START
        self._key_buffer = ""
        self.current_key: str | None = None
        self.current_value = ""
        self.selected_function_name: str | None = None
        self.object_stack: list[str | None] = [None]
        self._escaped = False
        self.constrained_engein: "ConstraintEngine" = constrained_engein

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self.state = JSONState.START
        self._key_buffer = ""
        self.current_key = None
        self.current_value = ""
        self.selected_function_name = None
        self.object_stack = [None]
        self._escaped = False

    def get_state(self) -> JSONState:
        """Get the current parsing state."""
        return self.state

    def update_state(self, token: str) -> None:
        """Advance the state machine based on the next token.

        Args:
            token (str): The next token text.

        Raises:
            _errors.FsmPredectionError: When the token is invalid for the current state.
            ValueError: When no more tokens are expected.
        """

        if token == '':
            return

        for char in token:
            self._update_state_char(char)

        self.constrained_engein.key_buffer = self._key_buffer
        self.constrained_engein.current_key = self.current_key
        self.constrained_engein.value_buffer = self.current_value
        self.constrained_engein.selected_function_name = self.selected_function_name


    def _update_state_char(self, char: str) -> None:
        if self.state == JSONState.START:
            self._handle_start(char)
        elif self.state == JSONState.OBJECT_START:
            self._handle_object_start(char)
        elif self.state == JSONState.EXPECT_NAME_KEY:
            self._handle_expect_name_key(char)
        elif self.state == JSONState.KEY_BODY:
            self._handle_key_body(char)
        elif self.state == JSONState.KEY_CLOSE:
            self._handle_key_close(char)
        elif self.state == JSONState.COLON:
            self._handle_colon(char)
        elif self.state == JSONState.VALUE_STRING_OPEN:
            self._handle_value_string_open(char)
        elif self.state == JSONState.VALUE_STRING_BODY:
            self._handle_value_string_body(char)
        elif self.state == JSONState.VALUE_STRING_CLOSE:
            self._handle_value_string_close(char)
        elif self.state == JSONState.VALUE_NUMBER:
            self._handle_value_number(char)
        elif self.state == JSONState.VALUE_OBJECT_CLOSE:
            self._handle_value_object_close(char)
        elif self.state == JSONState.COMMA:
            self._handle_comma(char)
        elif self.state == JSONState.OBJECT_END:
            if char.isspace():
                return
            raise ValueError("State machine is done — no more tokens expected.")
        else:
            raise ValueError(f"Unsupported JSON state {self.state!r}.")

    def _handle_start(self, char: str) -> None:
        if char.isspace():
            return
        if char == '{':
            self.state = JSONState.OBJECT_START
            return
        self._raise_prediction_error("'{'", char)

    def _handle_object_start(self, char: str) -> None:
        if char.isspace():
            return
        if char == '"':
            self._key_buffer = ""
            self.state = JSONState.EXPECT_NAME_KEY
            return
        self._raise_prediction_error('"', char)

    def _handle_expect_name_key(self, char: str) -> None:
        if char == '"':
            self.state = JSONState.KEY_CLOSE
            return
        self._key_buffer += char
        self.state = JSONState.KEY_BODY

    def _handle_key_body(self, char: str) -> None:
        if char == '"':
            self.state = JSONState.KEY_CLOSE
            return
        self._key_buffer += char

    def _handle_key_close(self, char: str) -> None:
        if char.isspace():
            return
        if char == ':':
            self.current_key = self._key_buffer
            self.state = JSONState.COLON
            if (self.object_stack[-1] == 'parameters' and
                self.current_key in self.constrained_engein.arguments):
                self.constrained_engein.arguments.remove(self.current_key)
            if (self.object_stack[-1] == 'parameters' and
                self.current_key not in self.constrained_engein.arguments):
                raise _errors.LLmModelError(f"Unknown argument: {self.current_key}")
            return
        self._raise_prediction_error("':'", char)

    def _handle_colon(self, char: str) -> None:
        if char.isspace():
            return
        if char == '"':
            self.current_value = ""
            self._escaped = False
            self.state = JSONState.VALUE_STRING_OPEN
            return
        if char == '{' and self.current_key == 'parameters':
            self.object_stack.append(self.current_key)
            self.current_key = None
            self.state = JSONState.OBJECT_START
            return
        if self._is_number_char(char):
            self.current_value = char
            self.state = JSONState.VALUE_NUMBER
            return
        self._raise_prediction_error("'INT' OR 'STR' or '{'", char)

    def _handle_value_string_open(self, char: str) -> None:
        if char == '"':
            self.state = JSONState.VALUE_STRING_CLOSE
            return
        self.current_value = char
        self._escaped = False
        self.state = JSONState.VALUE_STRING_BODY

    def _handle_value_string_body(self, char: str) -> None:
        if self._escaped:
            self.current_value += char
            self._escaped = False
            return
        if char == '\\':
            self.current_value += char
            self._escaped = True
            return
        if char == '"':
            if self.current_key == 'name' and self.object_stack[-1] is None:
                self.selected_function_name = self.current_value
                if self.selected_function_name not in self.constrained_engein.functions_name:
                    raise _errors.LLmModelError(f"Unknown function: {self.selected_function_name}")
                for f in self.constrained_engein.functions:
                    if f.name == self.selected_function_name:
                        self.constrained_engein.arguments = [arg.name for arg in f.parameters]
            self.state = JSONState.VALUE_STRING_CLOSE
            return
        self.current_value += char

    def _handle_value_string_close(self, char: str) -> None:
        if char.isspace():
            return
        if char == ',':
            self.current_key = None
            self.state = JSONState.COMMA
            return
        if char == '}':
            self._close_current_object()
            return
        self._raise_prediction_error("',' or '}'", char)

    def _handle_value_number(self, char: str) -> None:
        if char == ',':
            self.current_key = None
            self.state = JSONState.COMMA
            return
        if char == '}':
            self._close_current_object()
            return
        if self._is_number_char(char):
            self.current_value += char
            return
        self._raise_prediction_error("',' or '}'", char)

    def _handle_value_object_close(self, char: str) -> None:
        if char.isspace():
            return
        if char == ',':
            self.current_key = None
            self.state = JSONState.COMMA
            return
        if char == '}':
            self._close_current_object()
            return
        self._raise_prediction_error("',' or '}'", char)

    def _handle_comma(self, char: str) -> None:
        if char.isspace():
            return
        if char == '"':
            self._key_buffer = ""
            self.current_key = None
            self.state = JSONState.EXPECT_NAME_KEY
            return
        self._raise_prediction_error('"', char)

    def _close_current_object(self) -> None:
        if len(self.object_stack) > 1:
            self.object_stack.pop()
            self.current_key = None
            self.state = JSONState.VALUE_OBJECT_CLOSE
            return
        self.current_key = None
        self.state = JSONState.OBJECT_END

    def _is_number_char(self, char: str) -> bool:
        return char in '0123456789-.'

    def _raise_prediction_error(self, expected: str, got: str) -> None:
        msg = f"Wrong prediction by llm Expected {expected}, got {got!r}"
        raise _errors.FsmPredectionError(msg)

    def is_valid_token(self, token: str) -> bool:
        """Validating if this token should be next in the generating text."""
        saved_state = (
            self.state,
            self._key_buffer,
            self.current_key,
            self.current_value,
            self.selected_function_name,
            list(self.object_stack),
            self._escaped,
        )
        try:
            self.update_state(token=token)
            return True
        except (_errors.FsmPredectionError, ValueError):
            (
                self.state,
                self._key_buffer,
                self.current_key,
                self.current_value,
                self.selected_function_name,
                self.object_stack,
                self._escaped,
            ) = saved_state
            return False

    def __str__(self) -> str:
        return (
            f"JSONStateMachine("
            f"state={self.state.name}, "
            f"_key_buffer={self._key_buffer!r}, "
            f"current_key={self.current_key!r}, "
            f"selected_function={self.selected_function_name!r})"
        )
