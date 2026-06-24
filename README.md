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


### engine Algo for gen valid tokens
```python
tokenize prompt → input_ids
loop:
    logits = llm_sdk.get_logits_from_input_ids(input_ids)
    legal_ids = constraints.get_legal_tokens(current_state, schema)
    masked_logits = mask(logits, legal_ids)
    next_token = pick(masked_logits)
    input_ids.append(next_token)
    state_machine.advance(next_token_text)
until state_machine says "done"
```

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
```json
{   -> JsonState.START_OBJECT
  ":JsonState.START_KEY   ft_function:JsonState.STRING.  ":END_KEY : -> JsonState.COLON

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
10. [0.Prefix-Constrained Decoding](https://app.notion.com/p/prefix_decoding-ALL-types-388e6e3c12ea8036b40cdffbf3594174)
9. [Prefix-Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)

AI was used to refine this README structure, improve clarity, and ensure the required sections were covered in English. It was not used to replace the project logic or the constrained decoding design, which must be implemented in code.
