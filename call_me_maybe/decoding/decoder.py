"""Constrained decoding engine"""

import json
import torch

from call_me_maybe.llm.model import LLModel
from call_me_maybe.llm.logits import LogitsProcessor
from call_me_maybe.llm.vocab import VocabularyManager
from call_me_maybe.models.function import FunctionDefinition
from call_me_maybe.decoding.constraints import ConstraintEngine
from call_me_maybe.decoding.json_state_machine import JSONStateMachine


def token_texts_to_ids(llm: LLModel, token_texts: set[str]) -> set[int]:
    """Convert allowed token text strings into single token IDs.

    Only single-token texts are converted; multi-token strings are ignored.
    """

    allowed_token_ids: set[int] = set()
    for token_text in token_texts:
        token_ids_tensor = llm.encode_text(token_text)
        token_ids = token_ids_tensor[0].tolist()
        if len(token_ids) == 1:
            allowed_token_ids.add(token_ids[0])
    return allowed_token_ids


def generate_text(
        llm: LLModel,
        prompt: str,
        functions: list[FunctionDefinition],
        max_steps: int
    ) -> str:
    """Greedily generate text while printing each generation step."""

    input_ids = llm.encode_text(prompt)

    generated = input_ids[0].tolist()
    vocab_manager = VocabularyManager(llm.get_vocab_path())
    constrained_decoding = ConstraintEngine(
        functions=functions,
        vocab_manager=vocab_manager
    )
    json_state_machine = JSONStateMachine(
        constrained_engein=constrained_decoding
    )

    json_output = []
    for i in range(max_steps):
        #! Get the logits for all possible next tokens
        logits_for_next_token: list[float] = llm.get_all_next_tokens_logits(generated)
        logits_processor = LogitsProcessor(logits=logits_for_next_token)

        #! return allowed tokens based on the curr status
        best_token_id = logits_processor.get_best_token()
        current_best_token = llm.decode_text(input_ids.new_tensor([best_token_id]))

        allowed_token_texts = constrained_decoding.filter_logits(
            current_state=json_state_machine,
            current_best_token=current_best_token,
        )
        if allowed_token_texts is not None:
            allowed_token_ids = token_texts_to_ids(llm, allowed_token_texts)
            logits_processor.mask_logits(allowed_token_ids)

        #! Get the best next token and its score
        next_token_id: int = logits_processor.get_best_token()
        displayable_token = llm.decode_text(input_ids.new_tensor([next_token_id]))

        #! validate the next token with the json state machine and update the state if valid
        state = json_state_machine.is_valid_token(displayable_token)
        print(json_state_machine)
        if state is False:
            break
        best_10_token = logits_processor.get_top_k_tokens(10)
        for i in range(10):
            print(f'top {i} token: _{llm.decode_text(best_10_token[i])}_ | score: {logits_processor.get_token_score(best_10_token[i])}')
        print()
        print(f"STEP {i}")
        print(f"TOKEN ID: {next_token_id}")
        print(f"TOKEN SCORE: {logits_processor.get_token_score(next_token_id)}")
        print(f"CURRENT TEXT: {llm.decode_text(input_ids.new_tensor([generated]))}")
        token_in_vocab = vocab_manager.get_token_by_id(next_token_id)
        print(f"curr token in vocab: _{token_in_vocab}_")
        print(f"curr token display: _{displayable_token}_")
        generated.append(next_token_id)
        json_output.append(next_token_id)

    with open('function_calling_results.json', 'w', encoding='utf-8') as file:
        decoded_json = llm.decode_text(input_ids.new_tensor(json_output))
        json_obj = json.loads(decoded_json)
        json.dump(json_obj, file, indent=4)


    text = llm.decode_text(torch.tensor(generated))
    return text
