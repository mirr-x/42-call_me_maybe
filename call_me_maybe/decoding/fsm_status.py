"""State machine for validating simplified JSON token transitions."""

from call_me_maybe._types._types import JsonState


class JSONStateMachine:
    """Tracks parsing state while consuming JSON-like tokens."""

    def __init__(self) -> None:

        self.state = JsonState.START_OBJECT
        self.current_key = ""

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self.state = JsonState.START_OBJECT
        self.current_key = ""

    def get_state(self) -> JsonState:
        """Get the current parsing state."""
        return self.state

    def update_state(self, token: str) -> None:
        """ update the status of that next token could be based on the input token """

        if self.state == JsonState.START_OBJECT:
            if token == '{':
                self.state = JsonState.START

        if self.state == JsonState.START:
            if token == '"':
                self.state = JsonState.NAME
            if token == '}':
                self.state = JsonState.END

        # if self.state == JsonState.START_KEY:
        #     # FIXME: we can check here if it's really a valid func name
        #     self.state = JsonState.STRING

        # if self.state == JsonState.STRING:
        #     if token == '"':
        #         self.state = JsonState.END_KEY

        # if self.state == JsonState.END_KEY:
        #     if token == ':':
        #         self.state = JsonState.COLON




    # def is_token_valid(self, token: str) -> bool:
    #     """ check if this tokn is valid or not to be next in FSM """

    #     if self.state == JsonState.START_OBJECT:
    #         if not token:
    #             return 





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