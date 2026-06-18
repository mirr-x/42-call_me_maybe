_This project has been created as part of the 42 curriculum by molahrac._

# call me maybe

## Description

call me maybe is a function-calling project for large language models. The goal is to read a natural-language request, select the most relevant function from the available function catalogue, and generate a strictly valid JSON object containing the chosen function name and its required parameters.

The project is centered on constrained decoding. Instead of trusting the model to emit correct JSON on its own, the decoder filters the vocabulary token by token so that every generated sequence remains both syntactically valid JSON and compatible with the schema described in data/input/functions_definition.json.

The repository includes:

1. data/input/functions_definition.json, which describes the callable functions, their argument names, argument types, and return types.
2. data/input/function_calling_tests.json, which contains prompts used to validate the implementation.
3. data/output/function_calling_results.json, which is the final generated result file expected by the project.
4. llm_sdk, a small wrapper package used to interact with the provided model and vocabulary.

## Instructions

### Installation

Install the Python dependencies required by the project with:

```bash
make install
```

This installs the tooling used by the repository, including pytest, flake8, mypy, and pydantic.

### Execution

Run the implementation entrypoint from the repository root once your solution is in place and the dependencies are installed. The program must read the input files, generate the function-calling results, and write them to data/output/function_calling_results.json.

### Validation

Use the provided checks when developing:

```bash
make lint
```

You can also run the project tests if you add or maintain them in your solution.

## Algorithm Explanation

The core of the solution is schema-aware constrained decoding.

1. Load and validate the function definitions from data/input/functions_definition.json.
2. Read the prompts from the test input file.
3. Build the expected JSON structure for each answer: prompt, name, and parameters.
4. At each generation step, inspect the current partial output and determine which JSON tokens are allowed next.
5. Use the vocabulary JSON provided by the SDK to map token ids to their string forms, then mask out tokens that would break JSON syntax or violate the function schema.
6. Select only among the remaining valid tokens until the object is complete.
7. Parse and validate the final JSON before writing the output file.

This approach guarantees that the output remains machine-readable and that parameter types match the function specification instead of relying on post-processing or prompt formatting alone.

## Design Decisions

The implementation is designed around small, testable stages rather than one monolithic generation loop.

The function catalogue is treated as the source of truth. No function names or parameter layouts are hardcoded, which keeps the solution compatible with changing test files during peer review.

JSON handling is strict. Missing files, malformed input, or schema mismatches should be handled explicitly so the program fails safely instead of emitting partial or invalid output.

The decoder works with token ids rather than plain text. That choice makes it possible to validate every candidate token against the vocabulary and preserve exact control over the generated structure.

## Performance Analysis

Accuracy is the main benefit of this approach. Because invalid tokens are removed before selection, the resulting JSON is far more reliable than a prompt-only solution and is expected to remain valid even on difficult inputs.

Speed is slightly reduced compared with unconstrained generation because each token requires extra schema checks and vocabulary filtering. That overhead is acceptable for this project because correctness and validity matter more than raw throughput.

Reliability is high. The program does not depend on the model spontaneously formatting its answer correctly, and it can reject malformed input files early instead of producing unusable output.

## Challenges Faced

The hardest part is balancing JSON validity with schema validity at token level. A token can be syntactically legal in JSON but still produce an invalid value for the expected type, so the decoder must track both structure and semantics.

Another difficulty is numeric handling. The decoder must accept integers and floating-point values when the schema allows numbers, while still rejecting tokens that would create malformed literals.

Missing or invalid input files are also important. The implementation needs to survive absent JSON files and broken content without crashing in an uncontrolled way.

## Testing Strategy

Validation should cover both happy-path and failure-path behavior.

1. Compare generated output against known prompts and expected function selections.
2. Parse the produced file with a strict JSON parser to confirm there are no syntax errors.
3. Check that every result contains exactly the required keys and no extras.
4. Verify that argument values match the declared types in functions_definition.json.
5. Test malformed JSON input and missing-file scenarios to confirm graceful error handling.
6. Run linting and static checks with make lint during development.

## Example Usage

After installation, the project should be run from the repository root so it can access the input and output paths used by the subject.

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

The final program must write an array of such objects to data/output/function_calling_results.json.

## Resources

Reference material used for this project includes:

1. [Python json module documentation](https://docs.python.org/3/library/json.html)
2. [constrained decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)
3. [Pydantic documentation](https://docs.pydantic.dev/latest/)
4. [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
5. [The 42 subject PDF](en.subject.pdf)
6. [Function definitions input](data/input/functions_definition.json)
7. [Function calling test set](data/input/function_calling_tests.json)
8. [Main FSM engein for constrained decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output#from-schema-to-grammar)

AI was used to refine this README structure, improve clarity, and ensure the required sections were covered in English. It was not used to replace the project logic or the constrained decoding design, which must be implemented in code.
