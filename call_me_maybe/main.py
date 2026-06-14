""" Call Me Maybe Entry Point """

import logging

from call_me_maybe.parsers.parser import Parsing
from call_me_maybe.parsers import _errors
from call_me_maybe.llm.model import LLModel
from call_me_maybe.llm.vocab import VocabularyManager

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
        txt = "hello my name"
        encoded = llm.encode_text(txt)
        logging.info('encoded text %s', encoded)
        decoded = llm.decode_text(encoded)
        logging.info('decoded token ids tensor obj: %s', decoded)

        vocab_path = llm.get_vocab_path()
        vocabularymanager = VocabularyManager(vocab_path=vocab_path)
        logging.info('vocab path: %s\n', vocab_path)

        print()
        logging.info('str to id -> %s', vocabularymanager.get_id_by_token('hello'))
        logging.info('id to str -> %s', vocabularymanager.get_token_by_id(14990))
        logging.info('vocabulary_size: %s', vocabularymanager.vocabulary_size())

        # logging.info('tokenizer path: %s', llm.get_tokenizer_path())
        # logging.info('tokenizer meges: %s', llm.get_merges_path())

        logists = llm.get_all_next_token_logits(encoded.tolist())
        logging.info('logists len(): %s\n', len(logists))


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
