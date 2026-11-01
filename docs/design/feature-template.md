# Feature authoring guide

Use this guide for source and documentation work under the current vertical-slice
architecture. **Do not run tests, smoke checks, validation commands, project
imports, models, notebooks or dependency experiments.** That user instruction
supersedes generic development advice and historical test commands. No dependency
addition or runtime-flag change is part of this guide.

## Choose the owner before adding files

Place an operation beside the feature that owns its behavior and artifacts.
For example, hand projection preview belongs to hand motion, packed-output
admission belongs to submission, and a reconstruction accumulator belongs to
object mesh. Share a primitive only when its technical contract is stable and
independent of feature selection policy.

Do not create generic service/repository layers to move a feature's handler,
persistence and integration into separate architectural tiers. Private helpers
can separate complex code within a slice. Public cross-feature metadata, such as
campaign selections, should expose a narrow documented boundary.

## Typical slice structure

```text
track3/features/example/
  README.md             Purpose, contract, configuration and limitations
  __main__.py           Thin operation/--request dispatch
  operation.py          Request parsing and owning handler
  _integration.py       Optional backend adapter kept inside the slice
  _geometry.py          Feature-local computation when needed
  operation.json        Explicit request draft, unresolved inputs left null
  runtime.json          Execution flags defaulting to false
  backends.json         Optional pinned runtime/model contract when applicable
```

This is a pattern, not a requirement to create empty files. Match the existing
feature's conventions. A configuration key does not implement an operation;
expose only implemented handlers in the CLI.

## Request and authorization contract

Document the operation, split/episode scope, explicit parent run paths, algorithm
parameters, resource caps and output ownership. Reject unknown or inconsistent
input meaning at the owning boundary. Keep null unresolved fields visible rather
than guessing a producer.

Operation handlers must respect their existing authorization checks before
optional backend imports or writes. Preserve separate development, validation
and evaluation controls where present. A false flag cannot be enabled by a queue
plan. An enabled flag would still not override an explicit user prohibition.

Adding a new production dependency requires a concrete proposal and confirmation.
Name package/source revisions, assets, licence requirements, environment, storage
and integration purpose. Do not run resolution experiments while preparing the
proposal. The MoGe2 approval does not transfer to unrelated packages.

## Producer and consumer responsibilities

A producer owns an immutable run identity, request snapshot, explicit input and
source hashes, provenance, artifact inventory and useful failure record. A
consumer owns admission of the exact artifacts it uses. Reading a parent's
`run.json` is only part of that responsibility.

When changing a boundary, read its consumers as source and document:

- Required producer kind, completion status and allowed provenance.
- Artifact hash ownership and exact dataset/split/episode scope.
- Pixel grid, intrinsics, frame index and PTS agreement.
- Units, coordinate direction, body frame, object/hand ordering and rotation form.
- Shape/dtype/validity semantics, including NaN or zero fields where applicable.
- Whether old artifacts lack newly required fields and must be regenerated.

Do not mutate historical producer records to make them satisfy a new contract.
Do not downgrade diagnostic lineage or turn missing data into an observation.
When intentional completion is needed, the owning operation must record what it
filled and why.

## Failure and resource behavior

Distinguish malformed input, missing artifacts, ordinary measurement loss,
capacity overflow and interrupted execution. A partial array can be useful for
diagnosis while its failed run remains ineligible downstream. Never change status
to completed merely because a useful subset was written.

Keep array and metadata writes consistent with existing atomic-write patterns.
Store temporary files inside the agreed project storage. Bound accumulators and
large geometry queries deliberately; fail explicit limits instead of dropping
measurements silently. Document remaining unbounded/library working memory and
unmeasured costs. Source arithmetic cannot establish GPU fit or throughput.

## README outline

A feature README should answer these questions in a useful order:

1. What concrete result does the feature own, and what is its current source/run
   status? Which consumers depend on it?
2. Which operations exist, with what request/configuration and runtime boundaries?
3. What exact producer records and artifacts are consumed? Which clocks, frames,
   units, validity masks and provenance tags must agree?
4. What algorithm is implemented, which assumptions are deliberate, and which
   alternatives remain proposals?
5. What files and fields are produced, who owns their hashes, and how are failure
   and partial progress represented?
6. Which packages/assets are already approved, which are missing, and what
   resource limits are implemented versus unmeasured?
7. What evidence would be required after execution is authorized? What questions
   remain open? Do not present planned evidence as an existing result.

Link detailed local source readings where they support pinned upstream behavior.
Label historical measurements and proposed improvements clearly. Avoid copying
live deadlines, prices or service rules into an undated claim.

## Source-only change workflow

Read the relevant instructions and feature request/handler/consumer files.
Identify the narrow integration gap or documentation mismatch. Implement within
the authorized scope, keeping runtime flags false and existing user changes
intact. Read the edited prose/source as text to catch obvious inconsistencies;
do not turn that reading into imports, compile checks, linting, test execution,
link validation or a project operation.

Update the owning README, affected [artifact contracts](artifact-contracts.md),
[delivery ledger](version1.md) where scope/evidence changed, and the dated
[implementation status](../../reports/implementation-status.md). State exactly
what changed and what was not executed. If runtime evidence is required, leave
that requirement visible for a later authorized phase.

A documentation-only change needs no new tests. Under the current instruction,
source changes also receive no executed tests or validation. Do not describe
untested code as verified, compatible, accurate or production-ready.
