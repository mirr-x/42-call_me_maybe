"""Constrained decoding engine"""

import torch

from call_me_maybe.llm.model import LLModel
from call_me_maybe._types._types import JSONState
from call_me_maybe.llm.logits import LogitsProcessor
from call_me_maybe.llm.vocab import VocabularyManager
from call_me_maybe.models.function import FunctionDefinition
from call_me_maybe.decoding.constraints import ConstraintEngine
from call_me_maybe.decoding.json_state_machine import JSONStateMachine
from call_me_maybe.visualization.visualizer import GenerationVisualizer


def token_texts_to_ids(llm: LLModel, token_texts: set[str]) -> set[int]:
    """Convert token text strings to single-token IDs.

    Args:
        llm (LLModel): Language model tokenizer interface.
        token_texts (set[str]): Candidate token text values.

    Returns:
        set[int]: IDs for token texts that map to exactly one token.
    """

    allowed_token_ids: set[int] = set()
    for token_text in token_texts:
        token_ids_tensor = llm.encode_text(token_text)
        token_ids = token_ids_tensor[0].tolist()
        if len(token_ids) == 1:
            allowed_token_ids.add(token_ids[0])
    return allowed_token_ids


def _decode_token_id(
        llm: LLModel,
        token_id: int,
        tensor_template: torch.Tensor) -> str:
    """Decode a single token ID into text using a tensor template."""

    token_tensor = tensor_template.new_tensor([token_id])
    return llm.decode_text(token_tensor)


def _build_logits_processor(
        llm: LLModel,
        input_ids: torch.Tensor,
        generated: list[int]) -> LogitsProcessor:
    """Create a logits processor for the current generation step."""

    logits_for_next_token: list[float] = llm.get_all_next_tokens_logits(
        generated
    )
    return LogitsProcessor(
        logits=input_ids.new_tensor(logits_for_next_token, dtype=torch.float32)
    )


def _decode_current_best_token(
        llm: LLModel,
        logits_processor: LogitsProcessor,
        tensor_template: torch.Tensor) -> str:
    """Decode the highest-scoring token from the current logits."""

    best_token_id: int = logits_processor.get_best_token()
    return _decode_token_id(llm, best_token_id, tensor_template)


def mask_logits(
        llm: LLModel,
        logits_processor: LogitsProcessor,
        allowed_token_texts: set[str]) -> None:
    """Mask logits to allow only the provided token texts.

    Args:
        llm (LLModel): Model used to encode token texts to token IDs.
        logits_processor (LogitsProcessor): Processor whose logits are
            masked in place.
        allowed_token_texts (set[str]): Token text strings that should
            remain allowed.
    """
    allowed_token_ids = token_texts_to_ids(llm, allowed_token_texts)
    logits_processor.mask_logits(allowed_token_ids)


def generate_text(
        llm: LLModel,
        system_prompt: str,
        prefix_prompt: str,
        functions: list[FunctionDefinition],
        max_steps: int,
        visualizer: GenerationVisualizer | None = None
        ) -> str:
    """Generate a constrained JSON compltion for the provided system_prompt."""

    input_ids = llm.encode_text(system_prompt + prefix_prompt)

    generated = input_ids[0].tolist()
    vocab_manager = VocabularyManager(llm.get_vocab_path())
    constrained_decoding = ConstraintEngine(
        functions=functions,
        vocab_manager=vocab_manager
    )
    json_state_machine = JSONStateMachine(
        constrained_engein=constrained_decoding
    )
    json_state_machine.state = JSONState.COMMA

    if visualizer is not None:
        visualizer.start(system_prompt + prefix_prompt)

    json_output = []
    for _ in range(max_steps):
        # Build logits processor and decode the current best token
        logits_processor = _build_logits_processor(llm, input_ids, generated)
        current_best_token = _decode_current_best_token(
            llm, logits_processor, input_ids
        )

        # return allowed tokens based on the curr status
        allowed_token_texts = constrained_decoding.filter_logits(
            current_state=json_state_machine,
            current_best_token=current_best_token,
        )
        # mask the logits to only allow the allowed tokens
        if allowed_token_texts is not None:
            mask_logits(llm, logits_processor, allowed_token_texts)

        # validate the best token and update the state machine
        next_token_id: int = logits_processor.get_best_token()
        next_token_text = _decode_token_id(llm, next_token_id, input_ids)

        # bonus: if the best token is not valid, block it and get the next best token
        retried = False
        while json_state_machine.is_valid_token(next_token_text) is False:
            retried = True
            logits_processor.block_token({next_token_id})
            next_token_id = logits_processor.get_best_token()
            next_token_text = _decode_token_id(llm, next_token_id, input_ids)
        
        if visualizer is not None:
            allowed_count = (
                len(allowed_token_texts)
                if allowed_token_texts is not None
                else None
            )
            visualizer.step_event(
                state_name=json_state_machine.get_state().name,
                raw_best_token=current_best_token,
                allowed_count=allowed_count,
                final_token=next_token_text,
                was_masked=(current_best_token != next_token_text),
                retried=retried,
            )

        generated.append(next_token_id)
        json_output.append(next_token_id)

        if json_state_machine.get_state().name == 'VALUE_OBJECT_CLOSE':
            break

    final_text = llm.decode_text(input_ids.new_tensor(json_output))
    if visualizer is not None:
        visualizer.finish(prefix_prompt + final_text, success=True)

    return llm.decode_text(input_ids.new_tensor(json_output))