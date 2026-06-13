""" Call Me Maybe Entry Point """

import logging

from call_me_maybe.parsers.parser import Parsing
from call_me_maybe.parsers import _errors
from call_me_maybe.llm.model import LLModel

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
        # functions = parsing.get_function()
        # prompts = parsing.get_prompts()

        # Phase 2: Tokening
        llm = LLModel(model_name='Qwen/Qwen3-0.6B')
        txt = "hello world"
        print(llm.encode_text(txt))

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
