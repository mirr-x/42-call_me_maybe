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
            self, current_state: JSONStateMachine) -> set[str]:
        """Return a set of allowed next characters given the current JSON state.


        Args:
            current_state (JSONStateMachine): The current JSON state machine instance
            functions (list[FunctionDefinition]): List of function definitions to constrain against

        Returns:
            set[str]: Set of allowed next characters
        """

        state = current_state.get_state()
        leading_whitespace = {' ', '\n'}

        if state == JSONState.START:
            # current_state.update_state('{')
            return {'{'}
        elif state == JSONState.OBJECT_START:
            return {'"'}
        elif state == JSONState.KEY_OPEN:
            return set('abcdefghijklmnopqrstuvwxyz') #return {'name'} # magic shit
        elif state == JSONState.KEY_BODY: #we wont need it for now
            return {'"'}
        elif state == JSONState.KEY_CLOSE:
            return {':'}
        elif state == JSONState.COLON:
            # return {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '"'} after wanting full for now we only nee dname function
            return {'"'}
        elif state == JSONState.VALUE_STRING_OPEN:
            return set('abcdefghijklmnopqrstuvwxyz_')  # later change it to.  {'f', 'n', 'fn', 'a'}
        elif state == JSONState.VALUE_STRING_BODY:
            return set('abcdefghijklmnopqrstuvwxyz_"')
        elif state == JSONState.VALUE_STRING_CLOSE:
            return {',', '}'}
        elif state == JSONState.VALUE_NUMBER:
            return {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '"'}
        elif state == JSONState.COMMA:
            return {'"'}
        # elif state == JSONState.OBJECT_END:
        #     return leading_whitespace

        # default: no constraints
        return set()
