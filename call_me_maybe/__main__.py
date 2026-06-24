"""Call Me Maybe Entry Point"""

import logging

from call_me_maybe.models.function import FunctionDefinition
from call_me_maybe.llm.model import LLModel
from call_me_maybe.parsers import _errors
from call_me_maybe.parsers.parser import Parsing
from call_me_maybe.decoding.decoder import generate_text


FUNCTIONS_FILE = "data/input/functions_definition.json"
PROMPT_FILE = "data/input/function_calling_tests.json"

SYSTEM_PROMPT = """
You are a helpful assistant that generates JSON responses. You must use the exact function names by character as provided.
Include both the chosen function name and a parameters object in the output.

Available functions:
{functions_block}

User Query:
{prompt}

Response format:
{{"name": "<function_name>", "parameters": {{ ... }}}}

Response (JSON only):
"""


def creat_sys_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Create the prompt for the LLM to start generating."""

    functions_data = [f"- {f}\n" for f in functions]
    functions_block = "\n\n".join(functions_data)

    return SYSTEM_PROMPT.format(functions_block=functions_block, prompt=prompt)


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

        # prompt = 'Rip through a string and obliterate every pattern collision.'
        prompt = 'i need the square root of 16 and the square root of 25'
        llm = LLModel()
        system_prompt = creat_sys_prompt(prompt, functions)
        functions_name = [f.name for f in functions]
        final_text = generate_text(
            llm=llm,
            prompt=system_prompt,
            # functions=functions_name,
            max_steps=300
        )
        with open('function_calling_results.json', 'w+', encoding='utf-8') as f:
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