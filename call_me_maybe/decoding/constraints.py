""" fill me """

from call_me_maybe.decoding.state_machine import JSONStateMachine
from call_me_maybe._types._types import JsonState
from call_me_maybe.models.function import FunctionDefinition


class ConstraintEngine:
    """ fill me """

    def __init__(self) -> None:
        """ fill me """

    def get_allowed_tokens(
            self, current_state: JSONStateMachine, functions: list[FunctionDefinition]) -> set[str]:
        """ get all possible next token just by the cuurent token """

        state = current_state.get_state()

        if state == JsonState.START:
            current_state.update_state('{')
            return {'{'}

