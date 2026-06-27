"""Constraint engine for filtering logits based on JSON state and function definitions."""


from call_me_maybe.decoding.json_state_machine import JSONStateMachine
from call_me_maybe._types._types import JSONState
from call_me_maybe.models.function import FunctionDefinition
from call_me_maybe.llm.vocab import VocabularyManager


class ConstraintEngine:
    """Engine for constraining valid next tokens based on JSON parsing state."""

    def __init__(
            self,
            functions: list[FunctionDefinition],
            vocab_manager: VocabularyManager) -> None:
        """Initialize the constraint engine."""

        self.functions: list[FunctionDefinition] = functions
        self.functions_name = [f.name for f in self.functions]
        self.vocab_manager: VocabularyManager = vocab_manager

        self.key_options: list[str] = ['parameters', 'name', 'prompt']

        self.key_buffer = ""
        self.current_key: str | None = None
        self.value_buffer = ""
        self.selected_function_name: str | None = None

        self.arguments: list[str] | None = None

    def _get_valid_tokens_text_for_prefix_functions(self) -> set[str]:
        """ get all tokens that start with val_buffer """

        allowed = set()
        for token_text, _ in self.vocab_manager.token_to_id.items():
            temp_token = self.value_buffer + token_text
            for f in self.functions:
                if f.name.startswith(temp_token) and len(f.name) >= len(temp_token):
                    allowed.add(token_text)
        return allowed

    def _get_valid_tokens_text_for_prefix_arguments(self) -> set[str]:
        """ get all tokens that start with val_buffer """

        allowed = set()
        for token_text, _ in self.vocab_manager.token_to_id.items():
            temp_token = self.key_buffer + token_text
            for arg in self.arguments:
                if arg.startswith(temp_token) and len(arg) >= len(arg):
                    allowed.add(token_text)

        return allowed


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
            if self.key_options:
                return {self.key_options.pop()}
            elif self.arguments is not None:
                valid_token_text = self._get_valid_tokens_text_for_prefix_arguments()
                return valid_token_text
        elif state == JSONState.KEY_BODY:
            if '"' in current_best_token:
                return {'"'}
        elif state == JSONState.KEY_CLOSE:
            if ':' in current_best_token:
                return {':'}
        elif state == JSONState.COLON:
            if '{' in current_best_token:
                return {'{'}
            if '"' in current_best_token:
                return {'"'}
            if (current_best_token.lstrip('" ')).isdigit():
                return None # it being handeled on jsm.py
        elif state == JSONState.VALUE_STRING_OPEN:
            if self.current_key == 'name' and self.selected_function_name is None:
                valid_token_text = self._get_valid_tokens_text_for_prefix_functions()
                return valid_token_text
            return None
        elif state == JSONState.VALUE_STRING_BODY:
            if '"' in current_best_token:
                return {'"'}
            elif self.current_key == 'name':
                valid_token_text = self._get_valid_tokens_text_for_prefix_functions()
                return valid_token_text
            return None
        elif state == JSONState.VALUE_STRING_CLOSE:
            if ',' in current_best_token:
                return {','}
            return {',', '}'}
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
            return None

        return None
