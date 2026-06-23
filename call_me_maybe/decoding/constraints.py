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
            return {'"'}
        elif state == JSONState.EXPECT_NAME_KEY:
            return {'name'}
        elif state == JSONState.KEY_BODY:
            return {'"'}
        elif state == JSONState.KEY_CLOSE:
            return {':'}
        elif state == JSONState.COLON:
            return {'"'}
        elif state == JSONState.VALUE_STRING_OPEN:
            return None
        elif state == JSONState.VALUE_STRING_BODY:
            if '"' in current_best_token:
                return {'"'}
            return None
        elif state == JSONState.VALUE_STRING_CLOSE:
            return {',', '}'}
        elif state == JSONState.VALUE_NUMBER:
            return {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '"'}
        elif state == JSONState.COMMA:
            return {'"'}

        return None
