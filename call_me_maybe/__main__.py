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
Include the original prompt, the chosen function name, and a parameters object in the output.

Available functions:
{functions_block}

User Query:
{prompt}

Response format:
{{"prompt": "<user_query>", "name": "<function_name>", "parameters": {{ ... }}}}

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

        # pahse 2: LLm part
        llm = LLModel()
        for prompt in prompts:
            system_prompt = creat_sys_prompt(prompt.prompt, functions)
            final_text = generate_text(
                llm=llm,
                prompt=system_prompt,
                functions=functions,
                max_steps=300
            )

        with open('output_llm', 'w+', encoding='utf-8') as f:
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
