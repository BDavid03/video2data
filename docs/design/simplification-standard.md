# Code style

Write short, plain code. Example: the old `calibration/geometry.py` hand-rolled matrix
products, determinants and rigid inverses in pure Python and checked every scalar. We rewrote it
with NumPy, the same API and the same output keys, in two thirds of the lines. Do the same everywhere.

## Remove

- Hand-written linear algebra, statistics or image maths that NumPy, SciPy or OpenCV does in one call.
- Per-element type, finite or shape checks that NumPy or one boundary check already guarantees.
  Check once at the request boundary, then trust your own data.
- Re-hashing or re-reading the same file several times in one operation.
- Closures and helpers that only wrap one call. Functions over about 60 lines that mix checks,
  maths and file writes.
- Comments and docstrings that restate policy, history or permissions. Keep one short line of purpose.
- Rebuilding the same dict or JSON twice. Duplicate `_json` / `_integer` helpers when `src/` has one.
- Branches for cases the boundary check already rules out.

## Keep (other code depends on it)

- Public function and class names, request fields and defaults, CLI operation names, JSON keys,
  artifact paths and file names.
- Run flags (`runtime.json`) and fail-closed refusals. Bad input still raises, with the same
  exception type where callers catch it.
- Provenance: `run_fields`, `write_json`, `sha256_file`, `input_file`, `read_run`, and the
  `run.json` fields consumers read.
- `src/paths.py` for every path. No artifacts outside the checkout.
- Numbers: the same arrays and values, to floating-point tolerance.
- No new dependencies, no GPU or model runs, no notebooks.

## Verify

- `python3 -m py_compile <file>` for every edited file.
- `PYTHONPATH=. data/.runtime/linux-cuda/bin/python -c "import <module>"` for every edited module
  (CPU import only; do not run operations that write outputs).
- For a pure function, compare old and new outputs on a typical input in a throwaway script under
  `data/.runtime/tmp/`.
