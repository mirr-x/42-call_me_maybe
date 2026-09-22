"""Call Me Maybe Entry Point"""

import os
import json
import logging

from call_me_maybe.parsers import _errors
from call_me_maybe.llm.model import LLModel
from call_me_maybe.parsers.parser import Parsing
from call_me_maybe.decoding.decoder import generate_text
from call_me_maybe.models.function import FunctionDefinition
from call_me_maybe.visualization.visualizer import GenerationVisualizer


FUNCTIONS_FILE = "data/input/functions_definition.json"
PROMPT_FILE = "data/input/function_calling_tests.json"
OUTPUT_FILE = "data/output/function_calling_results.json"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
VISUALIZE = False

SYSTEM_PROMPT = """
You are a helpful assistant that generates JSON responses. You must use the
exact function names by character as provided. Include the original prompt, the
chosen function name, and a parameters object in the output.

Available functions:
{functions_block}

User Query:
{prompt}

Response format:
{{"prompt":"<user_query>","name":"<function_name>","parameters": {{ ... }}}}

Response (JSON only):
"""


def creat_sys_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Create the prompt for the LLM to start generating."""

    functions_data = [f"- {f}\n" for f in functions]
    functions_block = "\n\n".join(functions_data)

    return SYSTEM_PROMPT.format(functions_block=functions_block, prompt=prompt)


def extract_files_path_from_args() -> tuple[str, str, str, str, bool]:
    """Extract the file paths from command line arguments.

    Returns (functions_file, prompt_file, output_file, model_name, visualize).
    """
    arguments = os.sys.argv[1:]
    functions_file = FUNCTIONS_FILE
    prompt_file = PROMPT_FILE
    output_file = OUTPUT_FILE
    model_name = DEFAULT_MODEL
    visualize = VISUALIZE

    for i, arg in enumerate(arguments):
        if arg == "--functions_definition" and i + 1 < len(arguments):
            functions_file = arguments[i + 1]
        elif arg == "--input" and i + 1 < len(arguments):
            prompt_file = arguments[i + 1]
        elif arg == "--output" and i + 1 < len(arguments):
            output_file = arguments[i + 1]
        elif arg == "--model" and i + 1 < len(arguments):
            model_name = arguments[i + 1]
        elif arg == "--visualize":
            visualize = True

    return functions_file, prompt_file, output_file, model_name, visualize


def main() -> None:
    """Main function for Call Me Maybe"""
    logging.info("Call Me Maybe started\n")

    try:
        # extract arguments from command line
        (
            functions_file,
            prompt_file,
            output_file,
            model_name,
            visualize
        ) = extract_files_path_from_args()
        logging.info("Using functions definition file: %s", functions_file)
        logging.info("Using input prompt file: %s", prompt_file)
        logging.info("Using output file: %s", output_file)
        logging.info("Using model: %s", model_name)
        logging.info("Visualization enabled: %s", visualize)
        # Phase 1: Parsing
        parsing = Parsing(
            file_name_functions=functions_file, file_name_prompt=prompt_file
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
        llm = LLModel(model_name=model_name)
        json_output = []
        for prompt in prompts:
            prefix_prompt = f'{{"prompt": "{prompt.prompt}",'
            system_prompt = creat_sys_prompt(prompt.prompt, functions)
            visualizer = GenerationVisualizer(enabled=visualize)
            final_text = generate_text(
                llm=llm,
                system_prompt=system_prompt,
                prefix_prompt=prefix_prompt,
                functions=functions,
                max_steps=300,
                visualizer=visualizer
            )

            try:
                json_obj = json.loads(prefix_prompt + final_text)
            except json.JSONDecodeError:
                # if not valid JSON, store raw text
                json_obj = {"prompt": prompt.prompt, "raw": final_text}

            json_output.append(json_obj)

        output_directory = os.path.dirname(output_file)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=4)

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
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
