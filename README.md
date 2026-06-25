_This project has been created as part of the 42 curriculum by molahrac._

<img src="https://github.com/mirr-x/42-CC-1337/blob/main/images/call_me_maybe.png" alt="42 Porto Common Core Banner" />


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

Install dependencies and run:
```bash
uv sync
uv run python -m call_me_maybe --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

### Algorithm: Constrained Decoding Generation Pipeline

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

After installation, run from repository root:

```bash
make install
make run
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
```json
{   -> JsonState.START_OBJECT
  ":JsonState.START_KEY   ft_function:JsonState.STRING.  ":END_KEY : -> JsonState.COLON

```

The final program must write an array of such objects to data/output/function_calling_results.json.

## Project Status & Evaluation Report

**OVERALL STATUS: NOT READY FOR EVALUATION**

### CRITICAL Issues (Must Fix - Blocks Submission)

#### 1. ❌ Main Loop Only Processes ONE Prompt [call_me_maybe/__main__.py]
- **Issue**: Line 59 hardcodes a single prompt instead of iterating through all prompts from the input file
- **Current**: `prompt = 'Replace all occurrences matching a regex pattern in a string.'`
- **Required**: Loop through `prompts = parsing.get_prompts()` and process each one
- **Impact**: Program will fail on peer review - evaluators will test with different input files
- **Fix**: Implement loop to process all prompts and collect results in array

#### 2. ❌ Output Format is Broken [call_me_maybe/__main__.py]
- **Issue**: Output is written as raw text instead of valid JSON array
- **Current**: Writing `final_text` directly to file
- **Required**: Output must be valid JSON array: `[{prompt: "", name: "", parameters: {}}]`
- **Impact**: Output file cannot be parsed - 100% failure rate
- **Fix**: Serialize results to proper JSON format before writing to file

#### 3. ❌ Constraint Engine Incomplete [call_me_maybe/decoding/constraints.py]
- **Issue**: Multiple TODO comments indicating unfinished logic
- **Lines with TODO logic**:
  - Line 41: `EXPECT_NAME_KEY` state - not building key options from function definitions
  - Line 43: Not validating extracted keys against function parameters
  - Line 47, 51: `KEY_BODY` and `KEY_CLOSE` states return `None` (no constraints)
  - Line 79: `VALUE_OBJECT_CLOSE` returns `None` (incomplete)
- **Impact**: Constrained decoding is not enforced - defeats project purpose
- **Fix**: Complete constraint logic to enforce schema validation at each step

#### 4. ❌ Function Name Selection Not Implemented [call_me_maybe/decoding/constraints.py]
- **Issue**: Code doesn't select which function to call based on the prompt
- **Required**: LLM should choose from available functions in `functions_definition.json`
- **Current**: No logic to constrain function name generation to valid function names
- **Impact**: Generated JSON will have incorrect function names
- **Fix**: Add constraint for `name` field to only allow token sequences that form valid function names

#### 5. ❌ Missing Command-Line Interface [call_me_maybe/__main__.py]
- **Issue**: Program doesn't parse command-line arguments as required by PDF
- **Required Format**: 
  ```
  uv run python -m call_me_maybe [--functions_definition FILE] [--input FILE] [--output FILE]
  ```
- **Current**: Hardcoded file paths
- **Impact**: Moulinette and evaluators cannot specify custom input files
- **Fix**: Use `argparse` to parse `--functions_definition`, `--input`, `--output` arguments

#### 6. ❌ No Parameter Validation Against Schema [call_me_maybe/decoding/decoder.py]
- **Issue**: Generated parameters are not validated to match function definitions
- **Required**: Each parameter must have correct type (number vs string, required vs optional)
- **Current**: Decoder generates tokens but doesn't check if values match expected types
- **Impact**: Output parameters may have wrong types or missing values
- **Fix**: Validate generated parameters against FunctionDefinition schema before output

#### 7. ❌ JSONStateMachine Missing Nested Object Handling [call_me_maybe/decoding/json_state_machine.py]
- **Issue**: State machine doesn't properly handle nested JSON objects for complex parameter structures
- **Required**: Support `{"prompt": "...", "name": "...", "parameters": {...}}`
- **Current**: Incomplete handling of VALUE_OBJECT_CLOSE state
- **Impact**: Complex parameter objects will fail to parse
- **Fix**: Complete state machine transitions for nested JSON structures

### HIGH Priority Issues (Major Functionality Gaps)

#### 8. ❌ Constraint Engine Not Integrated with LLM Model [call_me_maybe/decoding/decoder.py]
- **Issue**: `filter_logits()` returns allowed tokens but integration is incomplete
- **Current Line 66**: Calls `constrained_decoding.filter_logits()` but doesn't always apply constraints
- **Problem**: When `allowed_token_ids` is empty or None, masking is skipped
- **Impact**: Model can generate invalid tokens, breaking JSON structure
- **Fix**: Ensure masking is always applied when constraints exist; handle edge cases

#### 9. ❌ No Error Handling for Decoding Failures [call_me_maybe/decoding/decoder.py + constraints.py]
- **Issue**: If state machine detects invalid token (line 69 breaks), no proper error recovery
- **Current**: Breaks loop silently, returns incomplete JSON
- **Required**: Graceful error handling with meaningful messages
- **Impact**: Silent failures make debugging impossible
- **Fix**: Add try-catch blocks and log error messages

#### 10. ❌ README Missing Mandatory Sections
- **Missing from PDF requirements**:
  - Algorithm explanation (constrained decoding approach)
  - Design decisions (why use state machine, etc.)
  - Performance analysis (accuracy, speed, reliability)
  - Challenges faced and solutions
  - Testing strategy and edge cases
- **Impact**: Evaluators cannot understand implementation
- **Fix**: Add comprehensive documentation sections

#### 11. ❌ No Support for Multiple Parameter Types [call_me_maybe/models/function.py + constraints.py]
- **Issue**: Types enum only has STRING and INTEGER, but PDF mentions `number` (float/double)
- **Current**: `Types.INTEGER` is used for both `integer` and `number`
- **Problem**: Cannot distinguish between 42 (integer) and 42.5 (float) in constraints
- **Impact**: Parameter type constraints will be incorrect
- **Fix**: Add FLOAT type to Types enum, update constraint logic

#### 12. ❌ Incomplete Logits Masking [call_me_maybe/llm/logits.py]
- **Issue**: The logits processor may not properly mask invalid tokens to negative infinity
- **Required**: Invalid tokens must have logits set to -inf to prevent selection
- **Impact**: Model can still generate forbidden tokens with low probability
- **Fix**: Verify masking implementation sets correct values

### MEDIUM Priority Issues (Quality & Edge Cases)

#### 13. ⚠️ No Timeout Mechanism for Generation [call_me_maybe/decoding/decoder.py]
- **Issue**: `max_steps=300` parameter exists but no actual timeout or max-steps enforcement in loop
- **Risk**: Could loop indefinitely on malformed input
- **Fix**: Enforce max_steps counter

#### 14. ⚠️ No Edge Case Handling
- **Unhandled cases**:
  - Empty prompts: `""`
  - Very long prompts
  - Special characters in strings: `"', \, ""`
  - Ambiguous prompts with multiple valid functions
  - Input files with invalid JSON
- **Fix**: Add validation and error handling for edge cases

#### 15. ⚠️ Unused Variable [call_me_maybe/__main__.py:63]
- **Issue**: `functions_name = [f.name for f in functions]` is created but never used
- **Fix**: Remove or use in constraint engine

#### 16. ⚠️ Incomplete Resource Cleanup
- **Issue**: File handles in parser may not be properly managed
- **Fix**: Use context managers for all file operations

### What's Working ✓

- **Parser Infrastructure**: `FunctionDefinition`, `Prompt` models with pydantic validation
- **JSON State Machine**: Well-structured state transitions (though incomplete)
- **LLM Wrapper**: Good abstraction over llm_sdk with proper error handling
- **Input Loading**: Correctly loads and validates JSON files
- **Encoding/Decoding**: Proper integration with model tokenizer

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


## Next Steps to Pass Evaluation

**Priority 1 (CRITICAL - Complete before submission):**
1. Implement command-line argument parsing (--input, --output, --functions_definition)
2. Add main loop to process all prompts from input file
3. Fix output to write valid JSON array format
4. Complete constraint engine logic (function name selection, parameter type validation)
5. Complete JSONStateMachine for nested objects

**Priority 2 (HIGH - Fix logic before submission):**
6. Add proper error handling and error messages
7. Integrate constraints throughout decoder pipeline
8. Add parameter type validation
9. Test with provided input files to verify output format

**Priority 3 (MEDIUM - Add before evaluation):**
10. Update README with algorithm explanation and design decisions
11. Add edge case handling
12. Add performance metrics and testing results
13. Document challenges and solutions
