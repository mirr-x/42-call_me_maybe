"""Call Me Maybe Entry Point"""

import logging

import torch
from call_me_maybe.llm.logits import LogitsProcessor
from call_me_maybe.llm.model import LLModel
from call_me_maybe.llm.vocab import VocabularyManager
from call_me_maybe.parsers import _errors
from call_me_maybe.parsers.parser import Parsing

FUNCTIONS_FILE = "data/input/functions_definition.json"
PROMPT_FILE = "data/input/function_calling_tests.json"


def generate_text(llm: LLModel, prompt: str, max_steps: int) -> str:
    """Greedily generate text while printing each generation step."""

    input_ids: list[int] = llm.encode_text(prompt)

    generated = input_ids[0].tolist()
    text = llm.decode_text(input_ids)
    vocab_manager = VocabularyManager(llm.get_vocab_path())
    for i in range(max_steps):
        logits_for_next_token: list[float] = llm.get_all_next_tokens_logits(generated)
        logits_processor = LogitsProcessor(logits=logits_for_next_token)
        next_token_id: int = logits_processor.get_best_token()
        top_5_token_ids = logits_processor.get_top_k_tokens(5)

        generated.append(next_token_id)
        print()
        print(f"STEP {i}")
        print(f"TOKEN ID: {next_token_id}")
        print(f"TOKEN SCORE: {logits_processor.get_token_score(next_token_id)}")
        print(f"CURRENT TEXT: {llm.decode_text(input_ids.new_tensor([generated]))}")
        token_in_vocab = vocab_manager.get_token_by_id(next_token_id)
        print(f"curr token in vocab: {token_in_vocab}")
        print(f"curr token display: {vocab_manager.get_display_token_by_id(next_token_id)}")
        print("TOP 5 CANDIDATES:")
        for candidate_token_id in top_5_token_ids:
            candidate_token = vocab_manager.get_display_token_by_id(candidate_token_id)
            candidate_score = logits_processor.get_token_score(candidate_token_id)
            print(f"{candidate_token_id:>6} -> {candidate_token:<20} -> {candidate_score}")

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

        llm = LLModel()
        final_text = generate_text(llm, "what is the sum of 1 and 2 ?", max_steps=30)
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
