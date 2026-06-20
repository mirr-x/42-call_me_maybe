"""State machine for validating simplified JSON token transitions."""

# TRANSITION TABLE
# state              | token seen     | next state
# --------------------------------------------------------
# START              | {              | OBJECT_START
# OBJECT_START       | "              | KEY_OPEN
# KEY_OPEN           | any word chars | KEY_BODY
# KEY_BODY           | "              | KEY_CLOSE
# KEY_CLOSE          | :              | COLON
# COLON              | "              | VALUE_STRING_OPEN
# COLON              | digit / -      | VALUE_NUMBER
# VALUE_STRING_OPEN  | any chars      | VALUE_STRING_BODY
# VALUE_STRING_BODY  | "              | VALUE_STRING_CLOSE
# VALUE_STRING_CLOSE | ,              | COMMA
# VALUE_STRING_CLOSE | }              | OBJECT_END
# VALUE_NUMBER       | digit / .      | VALUE_NUMBER  (self-loop)
# VALUE_NUMBER       | ,              | COMMA
# VALUE_NUMBER       | }              | OBJECT_END
# COMMA              | "              | KEY_OPEN
# OBJECT_END         | (nothing)      | DONE


from call_me_maybe._types._types import JSONState
from call_me_maybe.models.function import FunctionDefinition
from call_me_maybe.parsers import _errors


class JSONStateMachine:
    """Tracks parsing state while consuming JSON-like tokens."""

    def __init__(self) -> None:

        self.state = JSONState.START
        self.current_key = ""
        self._key_buffer = ""
        self.functions: list[FunctionDefinition] = None

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self.state = JSONState.START
        self.current_key = ""
        self._key_buffer = ""

    def get_state(self) -> JSONState:
        """Get the current parsing state."""
        return self.state

    def update_state(self, token: str) -> None:
        """ update the status of that next token could be based on the input token """

        token = (token.strip()).lstrip('\n')
        if token == '':
            return
        if self.state == JSONState.START:
            if token == '{':
                self.state = JSONState.OBJECT_START
            else:
                msg = f"Wrong prediction by llm Expected '{{', got {token!r}"
                raise _errors.FsmPredectionError(msg)

        elif self.state == JSONState.OBJECT_START:
            if token == '"':
                self._key_buffer = ""
                self.state = JSONState.KEY_OPEN
            else:
                msg = f"Wrong prediction by llm Expected '\"', got {token!r}"
                raise _errors.FsmPredectionError(msg)

        elif self.state == JSONState.KEY_OPEN:
            self._key_buffer += token
            self.state = JSONState.KEY_BODY

        elif self.state == JSONState.KEY_BODY:
            if token == '"':
                self.current_key = self._key_buffer  # register the key
                self.state = JSONState.KEY_CLOSE
            else:
                self._key_buffer += token

        elif self.state == JSONState.KEY_CLOSE:
            if token == ':':
                self.state = JSONState.COLON
            else:
                msg = f"Wrong prediction by llm Expected ':', got {token!r}"
                raise _errors.FsmPredectionError(msg)

        elif self.state == JSONState.COLON:
            if token == '"':
                # self.current_value = ""
                self.state = JSONState.VALUE_STRING_OPEN
            elif token.isdigit():  # token.lstrip("-").replace(".", "", 1).isdigit():
                self.state = JSONState.VALUE_NUMBER
            else:
                msg = f"Wrong prediction by llm Expected 'INT' OR 'STR' , got {token!r}"
                raise _errors.FsmPredectionError(msg)

        elif self.state == JSONState.VALUE_STRING_OPEN:
            # self.current_value += token
            self.state = JSONState.VALUE_STRING_BODY

        elif self.state == JSONState.VALUE_NUMBER:
            # self.current_value = int
            if token == ',':
                self.state = JSONState.COMMA
            elif token == '}':
                self.state = JSONState.OBJECT_END
            else:
                msg = f"Wrong int prediction by llm Expected ',' or '}}', got {token!r}"
                raise _errors.FsmPredectionError(msg)

        elif self.state == JSONState.VALUE_STRING_BODY:
            if token == '"':
                # self.current_value = self._key_buffer  # register the key
                self.state = JSONState.VALUE_STRING_CLOSE
            else:
                # self.current_value += token
                pass

        elif self.state == JSONState.VALUE_STRING_CLOSE:
            if token == ',':
                self.state = JSONState.COMMA
            elif token == '}':
                self.state = JSONState.OBJECT_END
            else:
                msg = f"Wrong prediction by llm Expected ',' or '}}', got {token!r}"
                raise _errors.FsmPredectionError(msg)

        elif self.state == JSONState.COMMA:
            if token == '"':
                self._key_buffer = ""
                self.state = JSONState.KEY_OPEN
            else:
                msg = f"Wrong prediction by llm Expected '\"' , got {token!r}"
                raise _errors.FsmPredectionError(msg)

        elif self.state == JSONState.OBJECT_END:
            self.state = JSONState.DONE

        elif self.state == JSONState.DONE:
            raise ValueError("State machine is done — no more tokens expected.")

    def is_valid_token(self, token: str) -> bool:
        """ validating if this token should be next in the generating text """

        try:
            self.update_state(token=token)
            return True
        except ValueError:
            return False

    def __str__(self) -> str:
        return (
            f"JSONStateMachine("
            f"state={self.state.name}, "
            f"_key_buffer={self._key_buffer!r})"
        )

# ### Algo
# ```python
# tokenize prompt → input_ids
# loop:
#     logits = llm_sdk.get_logits_from_input_ids(input_ids)
#     legal_ids = constraints.get_legal_tokens(current_state, schema)
#     masked_logits = mask(logits, legal_ids)
#     next_token = pick(masked_logits)
#     input_ids.append(next_token)
#     state_machine.advance(next_token_text)
# until state_machine says "done"
# ```