# Call Me Maybe - Logic Compliance Evaluation Report

Date: 2026-06-27
Scope: Logic and functional compliance only (linting intentionally skipped as requested)
Reference: Requirement text provided in chat for the project PDF (v1.5)

## Executive Result

Overall status: **NOT FULLY COMPLIANT**

Weighted logic score (approx): **41 / 100**

Decision: **Not ready for peer evaluation** due to multiple blocking logic mismatches with required behavior.

## Ranking Scale

- PASS: requirement is correctly implemented
- PARTIAL: some parts implemented, but important gaps remain
- FAIL: requirement missing or behavior contradicts the requirement

## Top Blockers (Logic)

1. CLI contract is missing (`--functions_definition`, `--input`, `--output`) -> FAIL
2. Program processes only one hardcoded prompt (`prompts[3]`) instead of all prompts -> FAIL
3. Output contract mismatch (wrong file path and wrong aggregation format) -> FAIL
4. Crash path exists in generation (`json.loads` without safe handling) -> FAIL
5. Dependency policy conflict (forbidden stack used via bundled llm_sdk: torch/transformers/huggingface) -> FAIL under strict reading

---

## Requirement-by-Requirement Ranking

## IV. Common Instructions (logic-focused)

### IV.1 General Rules

1. Python 3.10+ -> PASS  
Evidence: `pyproject.toml` requires `>=3.12`.

2. Graceful exception handling / no unexpected crashes -> PARTIAL  
Evidence:
- Good parser/model custom error handling in `call_me_maybe/parsers` and `call_me_maybe/llm/model.py`.
- But generation may still crash: `call_me_maybe/decoding/decoder.py` calls `json.loads(decoded_json)` with no local `try/except`; `__main__.py` catches only parser/model errors.

3. Proper resource management (context managers) -> PASS  
Evidence: file I/O uses `with open(...)` in loader and output writes.

4. Type hints for functions/returns/variables -> PASS (logic scope)  
Evidence: type hints are broadly present across modules.

5. Docstrings present -> PASS (logic scope)  
Evidence: docstrings exist in most classes/functions.

### IV.2 Makefile

1. `install` target -> PASS
2. `run` target -> PASS
3. `debug` target -> PARTIAL  
Evidence: `python -m pdb -m call_me_maybe.main` appears incorrect (entry module is `call_me_maybe` / `__main__.py`).
4. `clean` target -> PASS
5. `lint` target -> PASS (not validated here by execution, only presence/intent)

### IV.3.1 Additional Requirements

1. All classes use pydantic validation -> PARTIAL  
Evidence: data models do use pydantic (`FunctionDefinition`, `Parameter`, `Prompt`, `OutputModel`), but not all classes in the project are pydantic models (e.g., engine/state classes are plain Python).

2. Allowed packages (numpy/json) -> PASS

3. Forbidden packages rule (`pytorch`, `huggingface`, `transformers`, etc.) -> FAIL (strict interpretation)  
Evidence:
- `llm_sdk/pyproject.toml` depends on torch/transformers/huggingface-hub.
- `llm_sdk/llm_sdk/__init__.py` imports and uses these libraries directly.

4. Model requirement (Qwen/Qwen3-0.6B default) -> PASS  
Evidence: `LLModel` defaults to `Qwen/Qwen3-0.6B`.

5. Function choice must come from LLM (not heuristics) -> PARTIAL  
Evidence:
- Function name is generated and validated by FSM.
- But top-level key sequence is heuristically forced in constraint engine (`key_options.pop()`), and constraints are not consistently strict in all states.

6. Do not use private llm_sdk methods/attributes -> PASS  
Evidence: wrapper uses public methods (`encode`, `decode`, `get_logits_from_input_ids`, `get_path_to_vocab_file`).

7. Must work with `uv sync` setup -> PARTIAL  
Evidence: project has uv config and local source mapping, but runtime behavior still fails major functional requirements below.

8. Program must never crash unexpectedly and provide clear errors -> PARTIAL/FAIL  
Evidence: parser errors are clear; decoder JSON parse path can still crash unexpectedly.

### IV.3.2 Usage

Required command:  
`uv run python -m src [--functions_definition ...] [--input ...] [--output ...]`

Status -> FAIL  
Evidence:
- Project module name is `call_me_maybe` (acceptable adaptation if consistent), but flags are not implemented.
- `call_me_maybe/__main__.py` uses hardcoded paths and no argument parser.

---

## V. Mandatory Part

### V.1 Summary (natural language -> function call JSON)

Status -> PARTIAL  
Evidence:
- Pipeline attempts to produce `{prompt, name, parameters}` style JSON.
- But end-to-end contract for all prompts and required output file is not met.

### V.2 Input Files

Status -> PARTIAL  
Evidence:
- Reads function definitions and prompts from default input files.
- Missing support for CLI-overridden paths.
- JSON error handling in parser is good.

### V.3 LLM Interaction

#### V.3.1 LLM SDK usage
Status -> PASS

#### V.3.2 Generation pipeline
Status -> PARTIAL  
Evidence: iterative logits -> constraints -> token selection loop exists.

#### V.3.3 Constrained decoding understanding and enforcement
Status -> PARTIAL  
Evidence:
- Real masking exists (`mask_logits` sets invalid token logits to `-inf`).
- FSM exists and validates transitions.
- But constraints are incomplete in multiple states (`None` => unconstrained), and key order behavior is heuristic.

### V.4 Output File Format

Status -> FAIL  
Required output: single JSON array in `data/output/function_calling_results.json`, each item contains exactly `prompt`, `name`, `parameters`.

Observed behavior:
- Code writes generated text to `output_llm`.
- Decoder writes parsed object to root `function_calling_results.json`.
- Main path currently processes only one prompt.

### V.4.2 Validation Rules

Status -> PARTIAL/FAIL

- Valid JSON always parseable -> FAIL risk (can break before complete JSON, then `json.loads`)
- Keys/types match schema exactly -> PARTIAL (some checks exist via FSM and pydantic)
- No extra keys/prose -> PARTIAL (FSM tends to constrain shape)
- All required arguments present -> PARTIAL (arguments tracked, but not fully guaranteed in all paths)
- Types match -> PARTIAL (string/number simplified; number mapped to integer enum)

### V.5 Performance and Reliability

Status -> FAIL (not demonstrably met)

- 90%+ accuracy: not measured/proven in project outputs
- 100% valid JSON/schema compliance: not guaranteed due incomplete constraints and crash path
- Process under 5 minutes: not formally validated in reportable way
- Robust error handling: incomplete in decoder path

### V.6 Testing Implementation

Status -> FAIL/PARTIAL

- No clear automated test suite demonstrating required edge cases and contract validation.
- Current runtime behavior still violates core functional contract.

---

## VI. README Requirements (logic/documentation completeness)

Status -> PARTIAL

What is good:
- First italicized line present
- English README
- Basic description and instructions present
- Mentions constrained decoding and examples

Gaps:
- README claims behavior not matching current code (CLI flags/output contract)
- Performance analysis is not backed by reproducible measured results
- Testing strategy and validated edge-case evidence are insufficiently demonstrated

---

## VII. Bonus (optional)

Status -> Not evaluated for pass/fail (optional), but currently not strongly demonstrated by executable evidence.

---

## VIII. Submission/Peer Review Readiness

Status -> FAIL (current logic state)

Main readiness gaps:
1. Missing CLI argument support
2. Not processing full prompt list
3. Output file/path/shape mismatch
4. Incomplete constrained-decoding guarantees
5. Potential forbidden dependency conflict under strict rule interpretation

---

## Evidence Pointers (Key Files)

- Entry/runtime flow: `call_me_maybe/__main__.py`
- Decoder and output write behavior: `call_me_maybe/decoding/decoder.py`
- Constraint logic: `call_me_maybe/decoding/constraints.py`
- FSM logic: `call_me_maybe/decoding/json_state_machine.py`
- Parsing and JSON error handling: `call_me_maybe/parsers/input_loader.py`, `call_me_maybe/parsers/parser.py`, `call_me_maybe/parsers/validator.py`
- Type modeling: `call_me_maybe/models/function.py`, `call_me_maybe/_types/_types.py`, `call_me_maybe/models/prompt.py`
- Dependency policy risk: `llm_sdk/pyproject.toml`, `llm_sdk/llm_sdk/__init__.py`
- Project command targets: `Makefile`

---

## Final Ranking Summary

- PASS: 11
- PARTIAL: 13
- FAIL: 11

Interpretation: The project has a solid base architecture (parser/models/FSM/masking skeleton), but **fails key mandatory end-to-end logic requirements** for evaluation in its current state.
