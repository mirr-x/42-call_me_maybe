_This project has been created as part of the 42 curriculum by molahrac._

<img src="https://github.com/mirr-x/42-CC-1337/blob/main/images/call_me_maybe.png" alt="42 Porto Common Core Banner" />


# call me maybe

## Description

call me maybe is a function-calling project for large language models. The goal is to read a natural-language request, select the most relevant function from the available function catalogue, and generate a strictly valid JSON object containing the chosen function name and its required parameters.

The project is centered on constrained decoding. Instead of trusting the model to emit correct JSON on its own, the decoder filters the vocabulary token by token so that every generated sequence remains both syntactically valid JSON and compatible with the schema described in `data/input/functions_definition.json`.

The repository includes:

1. `data/input/functions_definition.json`, which describes the callable functions, their argument names, argument types, and return types.
2. `data/input/function_calling_tests.json`, which contains prompts used to validate the implementation.
3. `data/output/function_calling_results.json`, which is the final generated result file expected by the project.
4. `llm_sdk`, a small wrapper package used to interact with the provided model and vocabulary.

## Instructions

Install dependencies and run:

```bash
uv sync
uv run python -m call_me_maybe --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

You can also use the Makefile shortcuts:

```bash
make install
make run
```

## Algorithm

The generation loop is a constrained decoding pipeline:

```text
tokenize prompt + prefix -> input_ids
repeat until JSON object is complete:
    1. ask the LLM for next-token logits
    2. read the current JSON state
    3. compute the set of legal next token texts
    4. mask every illegal token
    5. pick the highest-scoring remaining token
    6. append the token and advance the state machine
```

In practice, the decoder combines a JSON state machine with schema-aware constraints. The state machine ensures structural correctness such as braces, keys, colons, commas, and closing quotes. The constraint engine narrows the token set so the model can only emit function names and argument keys that exist in the provided schema.

## Design Decisions

1. JSON is generated under control of a finite-state machine instead of being post-processed after free-form text generation.
2. Function names and argument names are validated against the input schema so the final object stays aligned with the available catalogue.
3. The implementation keeps the `prompt` field from the input and emits `name` and `parameters` in a fixed output shape.
4. The decoder works token by token, which makes it easier to reject invalid continuations early instead of repairing malformed output later.
5. The program writes a JSON array of results to the output file, one object per input prompt.

## Performance Analysis

The runtime is dominated by repeated LLM inference. For each generated token, the program requests the next-token logits, applies constraints, and selects one token. That makes the total cost roughly linear in the number of generated tokens, with an additional per-step vocabulary scan when building the allowed token set.

The practical cost is bounded by `max_steps=300`, which prevents runaway generations. The trade-off is that stronger constraints improve output validity but can add token-filtering overhead, especially when the engine checks prefixes against the full vocabulary.

The implementation prioritizes correctness over throughput. That is appropriate for this project because the main requirement is to produce valid function-call JSON, not to maximize tokens per second.

## Challenges Faced

1. Keeping JSON syntax valid while still letting the model choose the correct function name and parameter values.
2. Handling multi-token names and prefixes without allowing invalid partial strings.
3. Balancing strict schema enforcement with the need to leave the model enough freedom to complete the response naturally.
4. Recovering cleanly when generation produces malformed output instead of crashing the whole run.
5. Ensuring the final file contains an array of results, not a single object.

## Testing Strategy

The project is validated with the prompts in `data/input/function_calling_tests.json`. A full run checks that the CLI can load the function definitions, generate one output per prompt, and write valid JSON to `data/output/function_calling_results.json`.

The main verification steps are:

1. Run the CLI with the provided input files.
2. Confirm the output file is valid JSON and is an array.
3. Check that each result keeps the original prompt and includes `name` and `parameters`.
4. Use the sample prompts to smoke-test both numeric and string-transformation cases.

## Example Usage

Example input prompt:

```json
{
  "prompt": "What is the sum of 2 and 3?"
}
```

Example output object:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

The final program writes an array of objects like this to `data/output/function_calling_results.json`.

## Resources

Reference material used for this project includes:

1. [Python json module documentation](https://docs.python.org/3/library/json.html)
2. [Constrained Decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)
3. [Pydantic documentation](https://docs.pydantic.dev/latest/)
4. [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
5. [The 42 subject PDF](en.subject.pdf)
6. [Function definitions input](data/input/functions_definition.json)
7. [Function calling test set](data/input/function_calling_tests.json)
8. [Main FSM engein for constrained decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output#from-schema-to-grammar)
10. [0.Prefix-Constrained Decoding](https://app.notion.com/p/prefix_decoding-ALL-types-388e6e3c12ea8036b40cdffbf3594174)
9. [Prefix-Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
