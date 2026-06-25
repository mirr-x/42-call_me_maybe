"""Constraint engine for filtering logits based on JSON state and function definitions."""

from call_me_maybe.decoding.json_state_machine import JSONStateMachine
from call_me_maybe._types._types import JSONState
# from call_me_maybe.models.function import FunctionDefinition
# from call_me_maybe.llm.vocab import VocabularyManager

# VocabularyManager.mask()

class ConstraintEngine:
    """Engine for constraining valid next tokens based on JSON parsing state."""

    def __init__(self) -> None:
        """Initialize the constraint engine."""
        # a mutable set of key options that can be popped as they are used
        self.key_options: list[str] = ['parameters', 'name', 'prompt']

    def filter_logits(
            self, current_state: JSONStateMachine, current_best_token: str
            ) -> set[str] | None:
        """Return a set of allowed next token text strings given the current JSON state.


        Args:
            current_state (JSONStateMachine): The current JSON state machine instance
            current_best_token (str): The best token text

        Returns:
            set[str] | None: Set of allowed next token strings, or None if no constraints are applied.
        """

        state = current_state.get_state()

        if state == JSONState.START:
            return {'{'}
        elif state == JSONState.OBJECT_START:
            if '"' in current_best_token: # i just added
                return {'"'}
            return None
        elif state == JSONState.EXPECT_NAME_KEY:
            if self.key_options: # TODO: creat function to handle and validate
                return {self.key_options.pop()}
            return None  # TODO: break here i should pull out all args and then give them to ai to choose from 
        elif state == JSONState.KEY_BODY:
            if '"' in current_best_token:
                return {'"'}
            return None # TODO: llm free to choose i should do some thing here
        elif state == JSONState.KEY_CLOSE:
            if ':' in current_best_token:
                return {':'}
            return None  # TODO: llm free to choose i should do some thing here
        elif state == JSONState.COLON:
            if '{' in current_best_token:
                return {'{'}
            if '"' in current_best_token:
                return {'"'}
            if (current_best_token.lstrip('" ')).isdigit():
                return None # im here right now i thick i should return {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '"'}  but it being handeled on jsm.py
        elif state == JSONState.VALUE_STRING_OPEN:
            return None
        elif state == JSONState.VALUE_STRING_BODY:
            if '"' in current_best_token:
                return {'"'}
            return None
        elif state == JSONState.VALUE_STRING_CLOSE:
            if ',' in current_best_token:
                return {','}
            return {',', '}'}  # {'}'}  here is corect one 
        elif state == JSONState.VALUE_NUMBER:
            if ',' in current_best_token:
                return {','}
            if '}' in current_best_token:
                return {'}'}
            return {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        elif state == JSONState.COMMA:
            if '"' in current_best_token:
                return {'"'}
        elif state == JSONState.VALUE_OBJECT_CLOSE:
            return None # TODO: return } or , or ...

        return None
