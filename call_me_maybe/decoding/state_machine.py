"""State machine for validating simplified JSON token transitions."""

from call_me_maybe._types._types import JsonState


class JSONStateMachine:
    """Tracks parsing state while consuming JSON-like tokens."""

    def __init__(self) -> None:

        self.state = JsonState.START
        self.current_key = ""

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self.state = JsonState.START
        self.current_key = ""

    def get_state(self) -> JsonState:
        """Get the current parsing state."""
        return self.state

    def is_valid_transition(self, token: str) -> bool:
        """Checks if token is allowed in current state."""
        if self.state == JsonState.START:


    # def update_state(self, token: str) -> None:
    #     """Update state based on the current token."""
        # STEP 1: start JSON
        # if self.state == JsonState.START:


        # STEP 2: inside object → expect KEY


        # STEP 3: KEY → COLON


        # STEP 4: COLON → VALUE


        # STEP 5: VALUE → next KEY or END
