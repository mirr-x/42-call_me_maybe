_This project has been created as part of the 42 curriculum by molahrac._

<img src="https://github.com/mirr-x/42-CC-1337/blob/main/images/call_me_maybe.png" alt="42 Porto Common Core Banner" />

<img src="https://github.com/mirr-x/42-CC-1337/blob/main/gif/call____me__maybe.gif" alt="42 Porto Common Core Banner" />



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

### Specifying a different model (bonus)

The `--model` flag accepts any Hugging Face model identifier compatible with the `llm_sdk` interface. The default is `Qwen/Qwen3-0.6B`.

```bash
uv run python -m call_me_maybe --model Qwen/Qwen3-1.7B
```

Both `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B` have been tested and produce correct output with constrained decoding. Larger models tend to assign higher logits to sensible continuations, which means constraints are triggered less often, but the output schema remains identical.

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
    6. if the token is invalid for the state machine, mask it and retry
    7. append the token and advance the state machine
```

In practice, the decoder combines a JSON state machine with schema-aware constraints. The state machine ensures structural correctness such as braces, keys, colons, commas, and closing quotes. The constraint engine narrows the token set so the model can only emit function names and argument keys that exist in the provided schema.

The `prompt` field is pre-filled into the generation context before the LLM runs. This means the model never generates the prompt text — it is copied verbatim from the input and used as context so the model understands what function to call.

## Design Decisions

1. JSON is generated under control of a finite-state machine instead of being post-processed after free-form text generation.
2. Function names and argument names are validated against the input schema so the final object stays aligned with the available catalogue.
3. The `prompt` field is pre-filled in Python before generation starts, guaranteeing the original text is preserved exactly and avoiding tokenization issues with special characters such as embedded quotes.
4. The decoder works token by token, which makes it easier to reject invalid continuations early instead of repairing malformed output later.
5. When a token fails state machine validation, the decoder masks it and retries with the next best token rather than aborting. This provides error recovery without requiring backtracking.
6. Number generation is bounded to 15 digits to prevent runaway decimal expansion on models that assign high logits to digit continuations.
7. The program writes a JSON array of results to the output file, one object per input prompt.

## Performance Analysis

The runtime is dominated by repeated LLM inference. For each generated token, the program requests the next-token logits, applies constraints, and selects one token. That makes the total cost roughly linear in the number of generated tokens, with an additional per-step vocabulary scan when building the allowed token set.

The practical cost is bounded by `max_steps=300`, which prevents runaway generations. The trade-off is that stronger constraints improve output validity but can add token-filtering overhead, especially when the engine checks prefixes against the full vocabulary.

The implementation prioritizes correctness over throughput. That is appropriate for this project because the main requirement is to produce valid function-call JSON, not to maximize tokens per second.

Tested results on the provided 11-prompt test set:

| Model | Correct | Valid JSON |
|---|---|---|
| Qwen/Qwen3-0.6B | 11/11 | 11/11 |
| Qwen/Qwen3-1.7B | 11/11 | 11/11 |

## Challenges Faced

1. Keeping JSON syntax valid while still letting the model choose the correct function name and parameter values.
2. Handling multi-token names and prefixes without allowing invalid partial strings.
3. Preventing premature string termination — the model frequently assigned high logits to `"` mid-string, especially after tokens like `in` or `]`. Fixed by removing the constraint-engine quote-forcing logic and letting the state machine own quote termination.
4. Prompts containing embedded double quotes (e.g. `"Hello 34"`) broke the JSON template. Fixed by pre-filling the prompt field in Python before encoding, escaping quotes as `\"` in the generation context.
5. Number fields on larger models produced runaway digit sequences. Fixed by tracking digit count in the constraint engine and forcing a terminator after 15 digits.
6. Balancing strict schema enforcement with the need to leave the model enough freedom to complete the response naturally.

## Testing Strategy

The project is validated with the prompts in `data/input/function_calling_tests.json`. A full run checks that the CLI can load the function definitions, generate one output per prompt, and write valid JSON to `data/output/function_calling_results.json`.

The main verification steps are:

1. Run the CLI with the provided input files.
2. Confirm the output file is valid JSON and is an array.
3. Check that each result keeps the original prompt and includes `name` and `parameters`.
4. Use the sample prompts to smoke-test both numeric and string-transformation cases.
5. Run with `--model Qwen/Qwen3-1.7B` to verify multi-model support produces equivalent output.

## Example Usage

Default model:

```bash
uv run python -m call_me_maybe
```

Custom model:

```bash
uv run python -m call_me_maybe --model Qwen/Qwen3-1.7B
```

All options:

```bash
uv run python -m call_me_maybe \
  --model Qwen/Qwen3-1.7B \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

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

## Bonus Features

- **Multiple model support**: pass `--model <hf_model_id>` to switch models at runtime. Tested with `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B`.
- **Error recovery**: when the best token is invalid for the current JSON state, the decoder masks it and retries with the next best token instead of aborting generation.

## Resources

1. [Python json module documentation](https://docs.python.org/3/library/json.html)
2. [Constrained Decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)
3. [Pydantic documentation](https://docs.pydantic.dev/latest/)
4. [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
5. [The 42 subject PDF](en.subject.pdf)
6. [Function definitions input](data/input/functions_definition.json)
7. [Function calling test set](data/input/function_calling_tests.json)
8. [Main FSM engine for constrained decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output#from-schema-to-grammar)
9. [Prefix-Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
10. [Prefix-Constrained Decoding (Notion)](https://app.notion.com/p/prefix_decoding-ALL-types-388e6e3c12ea8036b40cdffbf3594174)
