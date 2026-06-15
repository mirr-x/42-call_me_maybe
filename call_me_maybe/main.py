""" Call Me Maybe Entry Point """

import logging

from call_me_maybe.parsers.parser import Parsing
from call_me_maybe.parsers import _errors
from call_me_maybe.llm.model import LLModel
from call_me_maybe.llm.vocab import VocabularyManager
from call_me_maybe.llm.logits import LogitsProcessor

FUNCTIONS_FILE = 'data/input/functions_definition.json'
PROMPT_FILE = 'data/input/function_calling_tests.json'


def main() -> None:
    """ Main function for Call Me Maybe """
    logging.info("Call Me Maybe started\n")

    try:
        # Phase 1: Parsing
        parsing = Parsing(
            file_name_functions=FUNCTIONS_FILE,
            file_name_prompt=PROMPT_FILE
        )
        parsing.run()
        functions = parsing.get_function()
        prompts = parsing.get_prompts()
        logging.info(
            "Loaded %d functions and %d prompts",
            len(functions),
            len(prompts),
        )

        # Phase 2: Tokenizing -----------------------------
        llm = LLModel(model_name='Qwen/Qwen3-0.6B')
        txt = "The capital of France is"
        encoded = llm.encode_text(txt)
        logging.info('encoded text %s', encoded)
        decoded = llm.decode_text(encoded)
        logging.info('decoded token ids tensor obj: %s', decoded)

        print("\nTesting Vocab section----------------------------------")
        vocab_path = llm.get_vocab_path()
        vocabulary_manager = VocabularyManager(vocab_path=vocab_path)
        logging.info('vocab path: %s\n', vocab_path)

        logging.info('str to id -> %s', vocabulary_manager.get_id_by_token('hello'))
        logging.info('id to str -> %s', vocabulary_manager.get_token_by_id(14990))
        logging.info('vocabulary_size: %s', vocabulary_manager.vocabulary_size())
        logits = llm.get_all_next_token_logits(encoded.tolist())
        logging.info('logits len(): %s\n', len(logits))

        print("\nTesting logist section----------------------------------")
        logist_procesing = LogitsProcessor(logits=logits)
        best_token_index = logist_procesing.get_best_token(logits)
        logging.info('best token indx is : %s', best_token_index)
        logging.info('best token score is : %s', logist_procesing.get_token_score(best_token_index))
        logging.info('best token val is : %s', vocabulary_manager.get_token_by_id(best_token_index))
        best_10 = logist_procesing.get_top_k_tokens(10)
        for i in best_10:
            print(vocabulary_manager.get_token_by_id(i))

    except _errors.ParserError as e:
        logging.error('PARSING ERROR: {%s} : cause {%s}', e, e.__cause__)
    except _errors.LLmModelError as e:
        logging.error('LLM ERROR: {%s} : cause {%s}', e, e.__cause__)

    print()
    logging.info("Program has Ended")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
        )

    try:
        main()
    except KeyboardInterrupt as e:
        logging.error("Program has been interrupted by the user: %s", e)
