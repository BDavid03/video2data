# Feature design template

Every feature README under `track3/features/<slice>/` follows this structure. A
section with nothing to say is kept and marked "None", so a missing answer is
visible.

```markdown
# <Feature name>

Status: design only | implemented, never run | run on <date> (link to run record).
One paragraph: what this slice turns into what, and why the entry needs it.

## Role in scoring
Which leaderboard metrics depend on this slice's output, and how an error here
shows up in the score. Cite the scorer.

## Operations
One row per operation the slice exposes: name, what it produces,
which runtime it needs, whether it uses the GPU.

## Inputs
Each input artifact: producer slice, path pattern, the fields read, and what the
slice does when the input is missing, partial or marked invalid.

## Method
The approach, in the order it executes. Name the models or algorithms, the
frames and clocks at each step, and the reason for each choice. Separate what
is fixed by the contract from what is this project's choice.

## Outputs
Path pattern, file formats, every field with dtype, shape, unit, frame and
validity rule. State what a consumer may assume and what it must check.

## Configuration and authorization
The request and runtime files the slice owns, their defaults, and the
authorization flag that keeps the operation disabled.

## Dependencies
Packages, model weights and licences. Mark each as already installed, approved,
or needing the user's confirmation before installation.

## Resources
Expected GPU, RAM and disk use, labelled as estimates until measured. How the
slice takes the shared GPU lock.

## Failure handling
What is recorded on failure, what partial results remain, and what downstream
slices see.

## Separation of public labels
Exactly which labelled data the slice may touch, and the provenance tag that
keeps label-assisted diagnostics out of the evaluation path.

## Acceptance evidence
What would have to be observed, once runs are authorized, before this slice
counts as working. No thresholds invented beyond those the scorer defines.

## Open questions
Each with the person or source that can resolve it.
```

## Rules for feature documents

- A feature owns its request, handler, integrations, persistence, configuration
  and documentation. It imports only the shared technical primitives
  (`track3/paths.py`, `track3/gpu.py`, `track3/provenance.py`) and reads other
  slices' artifacts from disk through the contracts in
  [artifact-contracts.md](artifact-contracts.md). It never imports another
  feature's private modules.
- A design document states no runtime, memory or accuracy figure as fact unless a
  recorded run produced it.
- New production dependencies are listed as proposals. Listing one is not
  approval to install it.
