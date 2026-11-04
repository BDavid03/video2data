# Feature template

How to add or change a slice (one folder under `src/features/`). Code style is in
[simplification-standard.md](simplification-standard.md).

## Pick the owner first

Put an operation in the slice that owns its behaviour and outputs. Share a helper only when its
contract is stable and has no feature policy in it. Do not add service or repository layers. Private
`_helpers.py` files inside a slice are fine.

## Typical slice

```text
src/features/example/
  README.md       What it does, how to run it, what it writes
  __main__.py     Thin dispatcher: <operation> --request <file.json>
  operation.py    Request parsing and the handler
  _integration.py Optional backend adapter, kept inside the slice
  operation.json  Request draft; unknown inputs stay null
  runtime.json    Run flags, false by default
  backends.json   Optional pinned model or runtime contract
```

This is a pattern, not a list of files to create. Expose only operations that exist.

## Requests and flags

A request names the operation, the episodes, explicit parent run paths, parameters, resource caps
and the output folder. Reject unknown or inconsistent fields at the boundary. Leave unknown inputs
null; never guess a producer.

Check the slice's `runtime.json` flags and episode allowlist before importing a backend or writing a
file. A queue file cannot turn a false flag on. Episode 41 and holdout episodes 43 and 44 are refused.

A new dependency needs a written proposal and the owner's yes: package and revision, assets, licence,
environment, storage and purpose.

## Producers and consumers

A producer writes a fixed run identity, a request snapshot, input and source hashes, provenance, an
artifact list and a useful failure record (see [artifact contracts](artifact-contracts.md)). A
consumer re-hashes the exact artifacts it uses; reading the parent's `run.json` is not enough.

When you change a boundary, check every consumer for: producer kind, status and allowed provenance;
hash ownership and episode scope; image grid, intrinsics, frame index and PTS; units, axes, body
frame, object and hand order, rotation form; array shape, dtype and validity (NaN or zero fields);
whether old outputs lack a new field and must be rebuilt.

Never edit old run records to fit a new contract. Never relabel diagnostic data or turn missing data
into an observation. If a step fills gaps, it records what it filled and why.

## Failures and resources

Tell apart bad input, missing files, normal measurement loss, capacity overflow and interruption. A
failed run may keep partial files for diagnosis but can never feed another run. Never mark a run
completed because part of it worked.

Write files atomically, as existing code does. Keep temp files inside the checkout. Bound buffers and
fail at the limit instead of dropping data. GPU jobs take the shared lease in `src/gpu.py`.

## Slice README

Answer, in order:

1. What does it do, and which pipeline step is it (see the README's "What we did")?
2. How do you run it (command, request file, flags)?
3. What does it read, and what must agree (frames, units, validity, provenance)?
4. What method does it use, and what does it assume?
5. What does it write, and where?
6. What are the known limits and open questions?

Label old measurements as old. Never present a planned result as a measured one.

## Check a change

- `python3 -m py_compile <file>` and a CPU import with the project interpreter.
- For pure functions, an old-versus-new comparison script under `data/.runtime/tmp/`.
- Update the slice README and [artifact contracts](artifact-contracts.md) in the same change.
