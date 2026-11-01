# Agent handoff for the research backlog

Read the repository-root `AGENTS.md` first, then `experiments/README.md`.
This file organises work; it grants no additional execution permissions.

- Work through the numbered queue in `README.md` in order. Start with the first
  unfinished eligible item and carry it through its recorded deliverables.
- Before changing anything, reconcile existing implementations and completed
  artifact records. Do not duplicate or restart the expanded learning campaign.
- Follow the prerequisites in each item. If an item is blocked, record the exact
  blocker and continue only with later work that does not depend on it. Return
  to the blocked item when its prerequisite is available.
- Every named approach needs a disposition: investigated, implemented,
  experimentally assessed, rejected with evidence, or deferred with a specific
  reason. A paper citation alone is not an experimental result.
- Keep implementation simple and feature-owned. Extend the existing vertical
  slices rather than introducing a generic experiment framework or new
  service/repository layers. Share only stable technical primitives.
- Record the hypothesis, baseline, changed variable, bounded resource budget,
  data split, intended metrics and stopping rule before an experiment.
- Preserve the root no-testing instruction. Verification methods described in
  this backlog are proposals, not permission to run tests, smoke checks,
  validation commands, sanitizers or dependency-resolution experiments.
- Use existing authorization for bounded research only where it actually
  applies. Do not infer permission for simulation, notebook execution, uploads,
  installations or new production dependencies from this backlog.
- Keep episode 41 excluded. Keep every episode's cameras, frames and derived
  artifacts in the same partition. Freeze choices before holdout outcomes.
  If an earlier campaign consumed the holdout, report that fact and do not reuse
  it as an untouched selection or final-assessment set.
- Keep runtime artifacts, environments, caches and temporary outputs under
  `Z:/hugging-face/video2data`. Existing external environments stay read-only.
- Keep diagnostic pose labels and reference scans out of production estimation.
- Update the queue and item notes with actual evidence, explicit failures and
  remaining work. Never turn a planned comparison into a claimed result.
- End each handoff with the last completed item, current item, next eligible
  item, artifact references and unresolved blockers. Do not mark the overall
  backlog complete while required items remain unexplored or unexplained.

These instructions are for future work in the main conversation. They do not
override the isolation or no-subagent rules of an ephemeral side conversation.
