"""State machine for validating simplified JSON token transitions."""

from call_me_maybe._types._types import JsonState


class JSONStateMachine:
    """Tracks parsing state while consuming JSON-like tokens."""

    def __init__(self) -> None:

        self.state = JsonState.START
        self.current_key = ''

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self.state = JsonState.START
        self.current_key = ''

    def get_state(self) -> JsonState:
        """Get the current parsing state."""
        return self.state

    def is_valid_transition(self, token: str) -> bool:
        """Checks if token is allowed in current state."""
        if self.state == JsonState.START:
            return token == "{"

        if self.state == JsonState.WAITING_FOR_KEY:
            return token not in (":", ",", "{")

        if self.state == JsonState.WAITING_FOR_COLON:
            return token == ":"

        if self.state == JsonState.WAITING_FOR_VALUE:
            return token not in (":", ",", "{")

        if self.state == JsonState.WAITING_FOR_NEXT:
            return token in (",", "}")

        return False

    def update_state(self, token: str) -> None:
        """Update state based on the current token."""
        # STEP 1: start JSON
        if self.state == JsonState.START:
            if token == "{":
                self.state = JsonState.WAITING_FOR_KEY
            return

        # STEP 2: inside object → expect KEY
        if self.state == JsonState.WAITING_FOR_KEY:
            if token == "}":
                self.state = JsonState.DONE
            else:
                self.state = JsonState.WAITING_FOR_COLON
            return

        # STEP 3: KEY → COLON
        if self.state == JsonState.WAITING_FOR_COLON:
            if token == ":":
                self.state = JsonState.WAITING_FOR_VALUE
            return

        # STEP 4: COLON → VALUE
        if self.state == JsonState.WAITING_FOR_VALUE:
            self.state = JsonState.WAITING_FOR_NEXT
            return

        # STEP 5: VALUE → next KEY or END
        if self.state == JsonState.WAITING_FOR_NEXT:
            if token == ",":
                self.state = JsonState.WAITING_FOR_KEY
            elif token == "}":
                self.state = JsonState.DONE
            return
