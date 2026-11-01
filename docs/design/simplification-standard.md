# Simplification standard (2026-10-29)

The user's direction: code like the old `track3/calibration/geometry.py` was
"ugly and over-engineered". That file hand-rolled matrix multiplication,
determinants and rigid inverses in pure Python, re-validated every scalar of
every row, and buried a schema check in a 100-line function. It was rewritten
with NumPy, the same public API and the same evidence keys, in two thirds of the
lines. Apply the same standard everywhere.

## Remove

- Hand-rolled linear algebra, statistics or image maths where NumPy/SciPy/OpenCV
  (already in the environment) do it in one call.
- Per-element type/finite/shape policing of values that NumPy coercion or one
  boundary check already guarantees. Validate once at the request boundary,
  then trust your own data.
- Re-hashing or re-reading the same file several times inside one operation
  "to detect tampering" between adjacent lines of the same process.
- Nested closures and helpers that exist only to wrap one call; functions over
  ~60 lines that interleave validation, computation and persistence.
- Docstrings and comments that restate policy, history or authorization prose
  instead of describing the code. Keep one short sentence of purpose.
- Repeated construction of the same dict/JSON; duplicated `_json`/`_integer`
  style helpers when `track3` already offers one.
- Branches for situations that cannot occur given the boundary validation.

## Keep (behaviour is a contract)

- Public function and class names, request dataclass fields and defaults,
  CLI operation names, JSON request/record keys and artifact paths/names.
- Authorization gates (`runtime.json` flags) and fail-closed refusals: an
  invalid input still raises, with the same exception type where consumers
  catch it.
- Provenance: `run_fields`, `write_json`, `sha256_file`, `input_file`,
  `read_run` usage and the `run.json` record fields consumers read.
- `track3.paths` for every path; no artifacts outside the permitted root.
- Numerical results: the same arrays/values to floating-point tolerance.
- No new dependencies, no GPU/model execution, no notebook execution.

## Verify

- `python3 -m py_compile <file>` for every edited file.
- `PYTHONPATH=. .runtime/linux-cuda/bin/python -c "import <module>"` for every
  edited module (CPU import only; do not run operations that write outputs).
- Where a function is pure, compare old vs new outputs on a representative
  input in a throwaway script under `.runtime/tmp/`.
