"""Call Me Maybe Entry Point"""

import logging
import torch

from call_me_maybe.models.function import FunctionDefinition
from call_me_maybe.llm.logits import LogitsProcessor
from call_me_maybe.llm.model import LLModel
from call_me_maybe.llm.vocab import VocabularyManager
from call_me_maybe.parsers import _errors
from call_me_maybe.parsers.parser import Parsing
from call_me_maybe.decoding.json_state_machine import JSONStateMachine
from call_me_maybe.decoding.constraints import ConstraintEngine


FUNCTIONS_FILE = "data/input/functions_definition.json"
PROMPT_FILE = "data/input/function_calling_tests.json"

SYSTEM_PROMPT = """
You are a helpful assistant that generates JSON responses. You must use the exact function names character by character as provided.

Available functions:
{functions_block}

User Query:
{prompt}

Response (JSON only):
"""


def get_allowed_tokens_ids(llm: LLModel, allowed_tikens: set[str]) -> set[int]:
    """Convert allowed token strings into token ids.

    Args:
        llm (LLModel): Language model used for tokenization.
        allowed_tikens (set[str]): Tokens allowed by the constraint engine.

    Returns:
        set[int]: Token ids corresponding to the allowed tokens.
    """

    valid_allowed_tokens = set()
    for token in allowed_tikens:
        encoded_token = llm.encode_text(token)
        valid_allowed_tokens.add(encoded_token.item())

    return valid_allowed_tokens


def creat_sys_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Create the prompt for the LLM to start generating."""

    functions_data = [f"- {f}\n" for f in functions for p in f.parameters]
    functions_block = "\n\n".join(functions_data)

    return SYSTEM_PROMPT.format(functions_block=functions_block, prompt=prompt)


def generate_text(llm: LLModel, prompt: str, max_steps: int) -> str:
    """Greedily generate text while printing each generation step."""

    input_ids = llm.encode_text(prompt)

    generated = input_ids[0].tolist()
    vocab_manager = VocabularyManager(llm.get_vocab_path())
    json_state_machine = JSONStateMachine()
    constrained_decoding = ConstraintEngine()
    for i in range(max_steps):
        logits_for_next_token: list[float] = llm.get_all_next_tokens_logits(generated)
        logits_processor = LogitsProcessor(logits=logits_for_next_token)
        allowed_tokens = constrained_decoding.filter_logits(current_state=json_state_machine)
        indices_masked_tokens = get_allowed_tokens_ids(llm, allowed_tokens)
        logits_processor.mask_logits(indices_masked_tokens)
        next_token_id: int = logits_processor.get_best_token()

        displayable_token = llm.decode_text(input_ids.new_tensor([next_token_id]))
        state = json_state_machine.is_valid_token(displayable_token)
        print(json_state_machine)
        if state is False:
            break
        print()
        print(f"STEP {i}")
        print(f"TOKEN ID: {next_token_id}")
        print(f"TOKEN SCORE: {logits_processor.get_token_score(next_token_id)}")
        print(f"CURRENT TEXT: {llm.decode_text(input_ids.new_tensor([generated]))}")
        token_in_vocab = vocab_manager.get_token_by_id(next_token_id)
        print(f"curr token in vocab: _{token_in_vocab}_")
        print(f"curr token display: _{displayable_token}_")
        generated.append(next_token_id)

    text = llm.decode_text(torch.tensor(generated))
    return text


def main() -> None:
    """Main function for Call Me Maybe"""
    logging.info("Call Me Maybe started\n")

    try:
        # Phase 1: Parsing
        parsing = Parsing(
            file_name_functions=FUNCTIONS_FILE, file_name_prompt=PROMPT_FILE
        )
        parsing.run()
        functions = parsing.get_function()
        prompts = parsing.get_prompts()
        logging.info(
            "Loaded %d functions and %d prompts",
            len(functions),
            len(prompts),
        )

        prompt = 'what is the sum of 1 and 2 ?'
        llm = LLModel()
        system_prompt = creat_sys_prompt(prompt, functions)
        final_text = generate_text(llm, prompt=system_prompt, max_steps=300)
        with open('model2.json', 'w+', encoding='utf-8') as f:
            f.write(final_text)
        logging.info("Final generated text: %s", final_text)

    except _errors.ParserError as e:
        logging.error("PARSING ERROR: {%s} : cause {%s}", e, e.__cause__)
    except _errors.LLmModelError as e:
        logging.error("LLM ERROR: {%s} : cause {%s}", e, e.__cause__)

    print()
    logging.info("Program has Ended")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        main()
    except KeyboardInterrupt as e:
        logging.error("Program has been interrupted by the user: %s", e)



# TODO: i have an aidia all works btw is just the llm predict more like i expected ':' and he gives ':"' it valid but not for my system what i need is give all to 10 predexctions and filter them by len then test them one by one 