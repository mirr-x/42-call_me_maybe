# Call Me Maybe - Logic Compliance Evaluation Report

Date: 2026-06-27
Scope: Logic and functional compliance only (linting intentionally skipped as requested)
Reference: Requirement text provided in chat for the project PDF (v1.5)

## Executive Result

Overall status: **PARTIALLY COMPLIANT**

Weighted logic score (approx): **66 / 100**

Decision: **Closer, but still not fully ready for peer evaluation**. The main end-to-end CLI/output blockers are fixed, but there are still remaining issues around dependency policy and workspace data availability.

## Ranking Scale

- PASS: requirement is correctly implemented
- PARTIAL: some parts implemented, but important gaps remain
- FAIL: requirement missing or behavior contradicts the requirement

## Verified Fixed Since Prior Review

1. CLI contract is implemented (`--functions_definition`, `--input`, `--output`).
2. The program now iterates over all prompts instead of only one hardcoded entry.
3. The output file is written as a JSON array to the requested path.
4. The JSON parse failure path is now handled in `main`, so malformed generations do not immediately crash the program.

## Remaining Blockers

1. Dependency policy conflict remains under strict reading because `llm_sdk` still depends on `torch`, `transformers`, and `huggingface-hub` -> FAIL.
2. The default workspace data folders are empty here, so running without external input files still fails in this checkout -> FAIL for bare `uv run python -m call_me_maybe`.
3. Constrained decoding still has heuristic and partially unconstrained branches -> PARTIAL.
4. `Makefile` debug target still points to `call_me_maybe.main` instead of the package entry point -> PARTIAL.

---

## Requirement-by-Requirement Ranking

## IV. Common Instructions (logic-focused)

### IV.1 General Rules

1. Python 3.10+ -> PASS  
Evidence: `pyproject.toml` requires `>=3.12`.

2. Graceful exception handling / no unexpected crashes -> PARTIAL  
Evidence:
- Good parser/model custom error handling in `call_me_maybe/parsers` and `call_me_maybe/llm/model.py`.
- The `main` entry point now catches JSON decode failures and falls back to raw output storage when generation is malformed.

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
Evidence: the project installs and the explicit CLI path works, but a no-argument run still fails in this checkout because the default data files are absent.

8. Program must never crash unexpectedly and provide clear errors -> PARTIAL  
Evidence: parser errors are clear and JSON parse failures are now caught in `main`, but decoding quality still depends on the model output.

### IV.3.2 Usage

Required command:  
`uv run python -m src [--functions_definition ...] [--input ...] [--output ...]`

Status -> PASS  
Evidence:
- Project module name is `call_me_maybe`, and the package entry point now accepts `--functions_definition`, `--input`, and `--output`.
- The verified execution path processes every prompt and writes the final JSON array to the requested output file.

---

## V. Mandatory Part

### V.1 Summary (natural language -> function call JSON)

Status -> PARTIAL  
Evidence:
- Pipeline now produces the required `{prompt, name, parameters}` JSON shape in the verified run.
- Universal correctness is still not formally proven across all model outputs.

### V.2 Input Files

Status -> PARTIAL  
Evidence:
- Reads function definitions and prompts from the supplied input paths.
- Default data folders are empty in this checkout, so the no-argument path still depends on external files being present.
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

Status -> PASS  
Required output: single JSON array in `data/output/function_calling_results.json`, each item contains exactly `prompt`, `name`, `parameters`.

Observed behavior:
- The CLI now writes the final JSON array to the requested output path.
- The current run with explicit paths produced one result per prompt.
- The previous hardcoded single-prompt behavior is no longer present.

### V.4.2 Validation Rules

Status -> PARTIAL/FAIL

- Valid JSON always parseable -> PARTIAL (fallback handling exists, but the model can still emit malformed output)
- Keys/types match schema exactly -> PARTIAL (some checks exist via FSM and pydantic)
- No extra keys/prose -> PARTIAL (FSM tends to constrain shape)
- All required arguments present -> PARTIAL (arguments tracked, but not fully guaranteed in all paths)
- Types match -> PARTIAL (string/number simplified; number mapped to integer enum)

### V.5 Performance and Reliability

Status -> PARTIAL

- 90%+ accuracy: not measured/proven in project outputs
- 100% valid JSON/schema compliance: not guaranteed due incomplete constraints
- Process under 5 minutes: not formally validated in reportable way
- Robust error handling: improved in the main execution path, but generation quality still depends on model output

### V.6 Testing Implementation

Status -> PARTIAL

- No clear automated test suite demonstrating required edge cases and contract validation.
- The main functional contract is now closer to the target, but the remaining gaps are still not fully covered by tests.

---

## VI. README Requirements (logic/documentation completeness)

Status -> PARTIAL

What is good:
- First italicized line present
- English README
- Basic description and instructions present
- Mentions constrained decoding and examples

Gaps:
- Performance analysis is not backed by reproducible measured results
- Testing strategy and validated edge-case evidence are insufficiently demonstrated

---

## VII. Bonus (optional)

Status -> Not evaluated for pass/fail (optional), but currently not strongly demonstrated by executable evidence.

---

## VIII. Submission/Peer Review Readiness

Status -> PARTIAL

Main readiness gaps:
1. Dependency policy conflict under strict rule interpretation
2. Default workspace data files are absent in this checkout
3. Incomplete constrained-decoding guarantees
4. Limited automated test evidence for the full contract

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

Interpretation: The project now satisfies the main CLI and output-path requirements in the current codebase, but it is still **not fully compliant** because the decoding stack remains only partially constrained and the bundled LLM SDK conflicts with the strict dependency rule.
