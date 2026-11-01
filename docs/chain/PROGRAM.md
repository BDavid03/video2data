# Chain program: Track 3 submission and recursive improvement

Every context in the chain reads this file first, after `AGENTS.md` and `CLAUDE.md`. It is the
standing brief. A hand-off prompt says where the last context stopped; this file says what the
work is and how it is done. Where a hand-off prompt and this file disagree, this file wins. Record
the disagreement in your hand-off so the owner can settle it.

The owner may edit this file at any time. Every context re-reads it, so an edit takes effect at
the next launch. The owner grants new permissions the way `AGENTS.md` already records them: a
dated line in `AGENTS.md`. Check `AGENTS.md` for new lines at the start of every context.

## 1. The finish line

The goal is the best honest Track 3 entry that can be submitted before the leaderboard freezes,
then a report that explains it well.

- **Leaderboard freeze: 4 November 2026, 17:00 EST (22:00 UTC; 09:00 AEDT on 5 November).**
  The driver prints the current time and the hours left in every context's system prompt. Use
  that clock, not a guess.
- **Package-ready target: 2 November 2026.** Anything after that is improvement, not rescue.

The deliverables, tracked in `docs/chain/SUBMISSION.md`:

| Id | Deliverable | Where the rules are |
| --- | --- | --- |
| D1 | Track 3 mesh file: one triangle mesh per `(episode, object)` in `track_3_evaluation_objects.json`, packed to Parquet, scored on CD-O | kit `tracks/track_3/README.md`, "Mesh submission" |
| D2 | Track 3 pose file: per-episode policy rollouts through the kit's `evaluate_policy.py`, packed to CSV, scored on AUC, SP-SR, MP-SR, RPE and MPPE | same file, "Pose submission" |
| D3 | Technical report: the two-page extended abstract in `templates/V2D_Challenge_Extended_Abstract_Template`, built to PDF. Award eligibility requires it | template `main.tex` header |
| D4 | Code release: the exact code at a GitHub `/commit/<sha>` URL, which both packed files must carry | kit `README.md`, "Shared setup" |
| D5 | Reproduction: one recorded command (or short script) per deliverable, with its run records | `CLAUDE.md`, run ownership |
| D6 | Packed files ready for the owner to upload, with a clean kit preflight apart from owner-only checks | kit `skills/v2d-submit/SKILL.md` |

The kit lives at `upstream/v2d_submission_kit/v2d_submission_kit/`. It is a read-only input.
Copy what you need to run into `outputs/` or `.runtime/`, never edit it in place.

**Triage rule.** A complete, valid, weaker submission beats an unfinished stronger one. Before
the freeze, work on the critical path of whichever deliverable is closest to submittable. If a
deliverable cannot be made by the freeze, prove it with measured evidence (what is missing, what
it would take), write that in `SUBMISSION.md`, and move the effort to what can still land.

What is known on 1 November 2026:

- The depth line is finished and frozen: `depth_student` iteration 17 (dilated, depth 8,
  256 px), tuning log-MAE 0.0756, holdout 43/44 log-MAE 0.0827 against MoGe2 pseudo-labels, ONNX
  exported with parity. See `docs/design/handoff-2026-10-29.md`.
- Every slice on the D1 and D2 paths (`object_masks`, `object_mesh`, `object_pose`,
  `scene_assembly`, `hand_motion`, `retargeting`, `policy_training`, `policy_rollout`) is
  implemented but has never run. Their `runtime.json` flags are mostly false. The owner's
  2026-10-18 line in `AGENTS.md` ("all flags are go, every experiment ran") lets a context turn a
  flag on. Record each flag you change, and why, in your hand-off. Episode 41 stays excluded and
  holdout 43/44 stays finalize-only, whatever a flag says.
- **Isaac Lab and FlashCHORD are approved for install** (owner, 2026-11-01). Neither is installed
  yet, and the venv has no `hydra`. Docker is present. Two risks to measure before committing to
  a route: Isaac Sim does not list WSL2 as a supported target, and the RTX 5080 (Blackwell)
  needs a recent Isaac Sim release. Compare the routes (a pip install in its own environment
  under `.runtime/`, the official container under Docker, or a native Windows install) with the
  cheapest probe that tells them apart, and record the result. FlashCHORD must run at
  `nvidia-isaac/video_to_data@aee95f25`, the revision the kit's evaluator matches. Installing
  Isaac Sim asks you to accept NVIDIA's licence. The owner's approval of the install is taken to
  cover that acceptance; say in the hand-off that it was accepted.
- **A LaTeX engine is approved** (owner, 2026-11-01): a reputable, established one from its
  official source, not one we write. The Tectonic release binary from
  `github.com/tectonic-typesetting/tectonic` (checksum checked), installed under
  `.runtime/tools/`, is the default; TeX Live is the fallback.
- **The owner uploads to Kaggle.** The chain produces the packed files under
  `submissions/<UTC stamp>/`, runs the kit preflight, and writes the exact upload commands into
  `NEEDS-YOU.md`. It never uploads and never accepts competition rules. The kit's own
  `requirements.txt` may be installed into a separate environment under `.runtime/kit-env/`,
  because packing needs it (an orchestrator reading of the owner's "ill upload to kaggle").
- **This checkout is a git repository** (initialised 2026-11-01, no remote). A GitHub
  `/commit/<sha>` URL still needs the owner to create the remote and push (`NEEDS-YOU.md`). Until
  then, pack with the local commit and mark the file as not final.
- Team name and authors do not matter yet: build the abstract with the template's `anonymous`
  class option.

## 2. What "recursive improvement across the board" means here

Every context leaves the project measurably better on at least one lens and no worse on any.
`docs/chain/SCOREBOARD.md` holds the lenses:

| Lens | What good looks like |
| --- | --- |
| Submission readiness | Every deliverable D1-D6 is either done with evidence or proven out of reach |
| Models | The depth, mask, mesh, pose and policy stages each have a measured best recipe, chosen on tuning data, with seeds and variance |
| Metrics | The official Track 3 metrics are computed locally where the kit allows (CD-O, pose metrics on public references), beside the depth metrics, with the pseudo-label caveat stated |
| Visualisations | Figures a reviewer understands without the text: qualitative depth and mesh samples, metric tables, error and failure maps, a pipeline diagram |
| Presentation | The extended abstract reads as a strong workshop paper: clear claim, method, results, limitations, within two pages |
| Insights | Ablations and failure analysis that explain why the numbers are what they are, not only what they are |
| Engineering | One command reproduces each deliverable; code follows `docs/design/simplification-standard.md`; nothing fails silently |

Each lens has a score from 0 to 5, the evidence for it, and the single next improvement.

**Who scores.** A lens score is set by a fresh-context critic, not by the context that did the
work. At the end of each context, launch one critic sub-agent (Opus 5.5 or Sonnet 5.5, as
`AGENTS.md` requires). Give it the scoreboard, the artifacts you changed and this section. Ask it
to score only the lenses you touched, with evidence, and to name the next improvement. Ask Codex
the same question as a second, cross-family critic (section 8). Where the two critics differ by
more than one point, the scoreboard takes the lower score and the hand-off says why they
differed. Keep your own disagreements with either critic in the hand-off. A score never rises
without new evidence.

**Choosing work.** In order:

1. Anything listed as running or half-finished in the newest hand-off.
2. Before the freeze: the critical path of the deliverable closest to submittable (triage rule).
3. The lowest-scoring lens whose next improvement can be measured inside this context.

Do one or two items per context and do them completely: run, measure, record, critic, commit,
hand off.

## 3. Strategy: the road less travelled

The owner's instruction (2026-11-01) is to take the road less travelled to get the best results.
Most entries will run the published reference pipeline as it stands. The edge is in what they
overlook. In practice:

**Read the scorer before building the model.** The kit ships code-only copies of every Track 3
metric in `metric_code/track_3/`. Work out exactly what each one rewards and what it ignores, and
aim each stage at that. Three examples from the Track 3 guide:

- CD-O registers each reconstructed mesh to the scanned mesh with its own Sim(3) transform before
  measuring the Chamfer distance. So a mesh's absolute scale and placement cost nothing; only its
  shape counts. Effort spent on metric scale for meshes is wasted. Effort on complete, closed,
  correctly proportioned shapes is not.
- The pose metrics align only the first scored pose of object slot 0, with one rigid transform
  per evaluation world. Scale error along the trajectory stays scored. So, unlike the meshes, the
  trajectories must be right in metric scale.
- MP-SR needs every object in an episode inside 3 cm and 30 degrees; SP-SR averages over
  objects; RPE is zero for a single-object episode. Know which scored episodes have one object,
  and where multi-object episodes lose their points.

**Name the alternative before taking the default.** For every stage, before running the default
route, write down at least one unconventional alternative with a plausibly larger payoff. Kinds
to consider: a different representation, a prior the others ignore, a closed-form step in place
of learning, per-episode test-time optimisation, a cheaper proxy that tracks the metric, or the
challenge's own one-policy-per-episode rule used deliberately. Estimate its cost, then run the
cheapest experiment that can tell it apart from the default. Prefer ideas that use structure in
this data: egocentric hand-object contact, the known object list per episode, static scene
geometry, consistency across the whole clip, and the depth student this project already owns.

**Keep the ideas ledger.** Record each idea in `docs/chain/IDEAS.md`: what it is, why it might
win, its cost, the cheapest test, and the result. An idea that was tried and lost is a finding.
It belongs in the insights lens and often in the report.

**Unconventional never means outside the rules.** Read the challenge rules (the challenge site
linked from the kit `README.md`, and `config/tracks.json`) before relying on any external data,
pretrained model or object asset. Record what they allow in `SUBMISSION.md`. Never use a
confidential reference, a holdout label, or anything a rule excludes.

## 4. Research integrity

These rules make the numbers believable. They are not optional, and they bind every context.

- Train, tuning and holdout are fixed: train 0/11/12/18/21/23/26/31, tuning 39/42, holdout
  43/44, `configs/splits/public-v1.json`. Episode 41 is excluded everywhere.
- Choose every model, recipe, threshold and figure on tuning data only. Holdout 43/44 was
  already assessed once, on 2026-10-29, and those numbers must not steer anything. At most one
  more `--finalize` may run in the whole chain: in the last context before the package is
  frozen, after `frozen-selection.json` is written. Report its numbers as they come out.
- Depth scores are against MoGe2 pseudo-labels, not measured depth. Every table and claim says so.
- Kaggle scores come back to the owner with a limited, rejection-counting quota. Record every
  packed file in `SUBMISSION.md` (file, hash, time), and the scores when the owner reports them.
  Never tune repeatedly against the leaderboard.
- Select by the usability gates plus the composite, never MAE alone (`auto_research/metrics.py`).
- Report variance across seeds wherever you report a model comparison.
- A result that looks surprisingly good gets one independent check before it is believed.

## 5. Procedure for each context

1. **Read**, in order: `AGENTS.md` (new owner lines first), `CLAUDE.md`, this file, the newest
   `docs/chain/handoffs/HANDOFF-*.md`, `docs/chain/SUBMISSION.md`, `docs/chain/SCOREBOARD.md`,
   `docs/chain/IDEAS.md`, `docs/chain/NEEDS-YOU.md`. Then say in one line what you will do.
2. **Check the machine.** Look at `ps`, `.runtime/gpu-stage.lock` and `git status`. Check the
   processes the hand-off lists as running. A run still going is waited on, never relaunched.
3. **Work** on the items chosen by section 2, with section 3 in mind. Ask Codex for its route
   and its best unconventional alternative before committing to one (section 8). Keep the vertical-slice
   pattern: request, gate, `run.json`, hashed inputs and artifacts. Long GPU work is started
   detached (`setsid nohup ... > outputs/<slice>/<name>.log 2>&1 &`), with its PID and log path
   written down at once.
4. **Measure** every change by running it on real data, and record where the evidence is.
5. **Critic** pass, with a Claude critic and a Codex critic, and the scoreboard update
   (sections 2 and 8).
6. **Commit** the finished item (section 6), then **hand off** (section 7) and end the turn.

**Context size.** Hand off after the second item, or at about 200k tokens of context, whichever
comes first. Never hand off in the middle of a step without writing down exactly where it is.
A long GPU run may keep going across the hand-off. Give its PID, its log and how the next
context can tell that it has finished.

**Nothing waits on the owner.** Every mechanics decision (schema, threshold, recipe, layout) is
yours: take the fail-closed default and record it. Only the items in section 6 need the owner.
Put those in `NEEDS-YOU.md`, in plain words with the exact action, and carry on with other work.

## 6. Safety rules

These come from `AGENTS.md`, `CLAUDE.md` and incidents in this project.

- Work only in `/home/ceolwolf/repos/video2data`. Never read or write `/mnt/z`. Keep every
  generated file, cache, tool and temp file under `outputs/`, `.cache/`, `.runtime/` or
  `submissions/`. The dataset snapshot, `upstream/` and `models/` are immutable.
- **Never `rm -rf` or `shutil.rmtree` anything except a directory you created yourself under
  `.runtime/tmp/`.** On 2026-10-29 sub-agents deleted `coursework/` and most of it was lost.
  Every sub-agent brief repeats this sentence.
- **Git.** Commit each finished item by explicit path (`git add <paths>`, then `git commit`),
  never `git add -A` or `git add .`, and keep large or generated files out (the `.gitignore`
  already excludes `outputs/`, `.runtime/`, `.cache/`, `models/`, `upstream/` and the dataset).
  Never rewrite history, force anything or touch a remote. Sub-agents run no mutating git; the
  context that owns the work commits it. Files git ignores are still copied to
  `.runtime/tmp/backup/<UTC stamp>/` before being rewritten, and a historical `run.json` is never
  changed.
- One GPU operation at a time, through the lock in `track3/gpu.py`. Never touch a live owner's
  lock, and never kill a process you did not start.
- No test suites. Verify the way `CLAUDE.md` describes: `py_compile`, a CPU import, an
  old-versus-new equivalence script for pure functions, and real runs as the evidence.
- **Approved installs** (owner, 2026-11-01): Isaac Lab, FlashCHORD and what they pull in; one
  reputable LaTeX engine; the kit's `requirements.txt` in its own environment. Install each in
  its own environment under `.runtime/`, record the exact versions and commands in the hand-off,
  and never modify the main `.runtime/linux-cuda` venv for them.
- **Owner-only, written to `NEEDS-YOU.md`, never done:** uploading to Kaggle or accepting
  competition rules; creating or pushing a GitHub repository; any other new dependency (write a
  concrete proposal so the owner can approve it in one line); anything that spends money.
- Sub-agents run on Opus 5.5 or Sonnet 5.5. Give each one its scope, the files and line ranges to
  read, the rmtree sentence, "no mutating git", and a tool-call budget.

## 7. The hand-off

Write `docs/chain/handoffs/HANDOFF-<UTC yyyymmddTHHMMZ>-<NN>.md`, where `NN` is your context
number. Write it last, in one write: the driver starts the next context once this file appears
and stops changing. Keep `SUBMISSION.md`, `SCOREBOARD.md`, `IDEAS.md` and `NEEDS-YOU.md`
current, and commit them with the hand-off.

Sections, in this order:

1. `# Hand-off NN: <one line on what this context did>`
2. `## Findings` — what was learned, with numbers and artifact paths. Include a `### Codex`
   list: each call's label and answer path, what it found, and what you took or declined and
   why. End it with the call count for this context and for the chain (`wc -l
   outputs/chain/codex/calls.jsonl`).
3. `## State` — a table: item, state, evidence path, next step.
4. `## Running` — every process still running: PID, command, log, how to tell it has finished.
5. `## Changed` — commits (hashes), runtime flags turned on, installs made.
6. `## Learned` — short lessons that bind later contexts.
7. `## Prompt for the next context (paste as the first message)` — last, and nothing after it.
   A self-contained prompt of at most about 6,000 characters. It names this hand-off file, the
   item to start with, and any fact the next context must not re-derive.

**Stopping the chain.** Put a line `CHAIN: STOP <one-line reason>` anywhere in the hand-off when
one of these holds:

- every deliverable is done or proven out of reach, and every lens scores at least 4;
- the freeze has passed, the report is final, and nothing useful remains;
- everything left waits on the owner.

Still write the prompt section, so the owner can restart the chain after answering `NEEDS-YOU.md`.

## 8. Collaborating with Codex

The owner asked for Codex collaboration in every context (2026-11-01). A second model family sees
different routes and different flaws, which is exactly what the road less travelled needs. Codex
is a collaborator, not a rubber stamp: weigh what it says against the evidence, and write down
where you disagree and why.

**One helper, always.** Call Codex only through
`bash scripts/chain/codex_ask.sh <label> <brief.md> [read-only|workspace-write] [model] [effort]`.
Never hand-roll a `codex exec` command. The helper adds the safety rules to every brief, keeps
the brief, answer and log under `outputs/chain/codex/`, and logs each call in
`outputs/chain/codex/calls.jsonl`. Launch it with `run_in_background`, end the turn, and read the
answer when you are woken. Never poll it. Write each brief to a file under `.runtime/tmp/` first.
Codex starts with no context, so the brief carries the facts, numbers, file paths and line ranges
it needs, and says what has already been ruled out.

**What every context does with Codex, at least:**

1. **Route and ideas** (read-only, `gpt-6-astra` high). Before committing to a stage's route, send
   Codex the stage, the measured facts, your intended route and the metric it is scored on. Ask
   for its preferred route, its best unconventional alternative with the cheapest test that would
   separate the two, and the step most likely to fail. Each new idea goes in `IDEAS.md` with
   "Codex" as its source.
2. **Critic** (read-only, `gpt-6-astra` high). At the end of the context, Codex scores the lenses
   you touched, beside the Claude critic (section 2).

**Where Codex can carry more:**

- **Review before a commit that changes behaviour** (read-only): give it `git diff` of the change
  and ask for defects only.
- **A bounded job in parallel** (workspace-write, `gpt-6.1-sol` high): one item with exclusive
  paths that nothing else in this context touches, a definition of done ("files written, run on
  real data, numbers reported"), and the files it may edit. You then review `git diff -- <paths>`,
  run its result yourself, and commit it. Codex never commits. Never give Codex and a Claude
  sub-agent the same paths, and never let a Codex job start GPU work while you hold the GPU.
- **Report and figure review** (read-only): a cold reading of the extended abstract or a figure,
  asking what a workshop reviewer would attack first.

**Budget.** Codex usage is limited. Plan about eight calls per context, and say the running count
in your hand-off. If an answer is empty and the log shows a usage limit, carry on with Claude
alone, record it under `### Codex`, and let the next context try again.
