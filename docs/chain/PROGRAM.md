# Chain program: Track 3 submission and recursive improvement

Every context in the chain reads this file first, after `AGENTS.md` and `CLAUDE.md`. It is the
standing brief. A hand-off prompt says where the last context stopped; this file says what the
work is and how it is done. Where a hand-off prompt and this file disagree, this file wins. Record
the disagreement in your hand-off so the owner can settle it.

The owner may edit this file at any time. Every context re-reads it, so an edit takes effect at
the next launch. The owner grants new permissions the way `AGENTS.md` already records them: a
dated line in `AGENTS.md`. Check `AGENTS.md` for new lines at the start of every context.

## 0. Short-term focus (owner, 2026-11-03 13:58 UTC): fundamentals before details

**The target is `docs/chain/COMPLETION-PLAN.md` (owner, 2026-11-04 about 01:40 UTC: "you still havent even got a model
trained or submission written to improve on, why have you ended the chain"). Its section 3 overrides section 7 below: the
chain does not stop for owner-only items while a workstream is open.**

**Owner's answers to that plan's section 7, 2026-11-04 01:53 UTC ("deadline stays the same, 2. yes, 3. yes. 4. tes"):**
1. Last pack 12:00 UTC, deadline 17:00 UTC, unchanged.
2. Scans may be used beyond measuring: to train a rotation model and to extend the H3 heading method past episode 19. Still no
   scan geometry in a submitted mesh or a simulated asset (the FAQ's reconstruction line); disclose every scan use in the abstract.
3. **The chain may push and upload**, overriding section 1 and section 6 on those two only. Push fast-forward only, never force:
   `git push origin worktree-chain-ctx01:main`, from the worktree. Upload only a file that passed its gate and with
   `check_release.sh` READY for it, once per pack; record file, hash, time and scores in `SUBMISSION.md`. Accepting competition rules
   and creating the Kaggle token stay the owner's. Scores still steer nothing (section 4).
4. FoundationPose install approved, in its own venv under `.runtime/`, with its licence recorded in `INSTALL.md` and the abstract.

Until the 4 November deadline, **structural improvements to the fundamentals come before refining the current
entry.** The owner's words: "please take bigger structural improvements before really refining the details", "e.g.
pretraining models, synthetic data generation, we're talking substantial improvements to the fundamentals". The
prompt was a leaderboard leader at pose AUC 0.99 against our static entry (train AUC 0.277; the oracle with true
frame-0 poses held static reaches only 0.532, so about half of AUC is lost to object motion we never reproduce).

- **Prefer:** objects that move along video-derived trajectories in the policy rollouts (per-frame tracking, hand
  motion, policies trained on moving references); whole-clip reconstruction instead of primitive meshes; pretrained
  foundation models (trackers, segmenters, image-to-3D) used as tools; synthetic data generated from our own
  reconstructions to train or adapt models; pretraining or adapting our own models on that data.
- **Defer:** tie-breakers, wording polish, further static-placement variants, unless they finish a pack.
- **Order of work:** the cheapest decisive experiment first (an upper bound that says whether a lever can pay at all),
  then the build. Go/no-go is measured on train/tuning under a protocol written first (section 4 unchanged).
- New packages or model weights still need a one-line proposal in `NEEDS-YOU.md` and a dated owner line in
  `AGENTS.md`; until then use what is installed or approved (section 6). Packs still finish by 12:00 UTC 4 Nov.

**Short-term to-do (owner list; tick here when done, with the commit):**

1. [x] Decisive upper bound: can our FlashCHORD policy follow a moving reference (train mocap, one train episode,
   16 GB GPU)? Evidence so far: `docs/chain/literature-structural.md`. *Context 19: D0 NO-GO (`beb640b`), D1 NO-GO on
   completion, transport unmeasured (`0eec3f3`); route paused for this deadline as a planning decision.*
2. [ ] If yes: per-frame object trajectories from released video (stereo depth + masks + own reconstruction), then
   policies on moving references for the scored episodes. If no: record the measured finding and move to item 3.
3. [~] Whole-clip mesh reconstruction (multi-frame fusion) in place of primitives, judged on CD-O train/tuning.
   *Context 20: F1 static pre-contact fusion KILL before any score (`4be1a52`): no second frame registered in the 0.6 s before
   contact (rough motion 2.4-5.2 deg); post-contact fusion needs masks through the clip (SAM2, unapproved). G1 (pretrained shapes) awaits approval. Context 21: F2 post-contact fusion routed NO-GO for this deadline (IDEAS 59).*
4. [~] Pretrained models and synthetic data where they serve 1-3 (proposals in `NEEDS-YOU.md` for any new install).
   *Context 20: Hunyuan3D-2mini installed on owner approval; G1 pilot FAIL (rack), G2 pot + pitcher PASS and packed as
   D1 v5 (`b161723`). SAM2.1 / grounding-dino / MediaPipe approved but not installed. Context 21: SAM2.1 and Grounding DINO installed for G3, which failed its packed veto (protocol `7bc865b`); MediaPipe not installed.*
5. [x] *(phases 1-2 `907bf80`, `4caf513`; phases 3-5 `f16821d`, `ea01ab6`, `a5ff859`; plan `docs/chain/restructure-plan.md`)* **Restructure the file system to be less scattered, and remove irrelevant components** (owner, 2026-11-03
   14:15 UTC). Order: (a) a written inventory first (each top-level directory, each slice, each script: used by a
   deliverable, by a recorded result, or by nothing), committed as a plan with the target layout; (b) moves with
   `git mv` only; nothing is deleted: everything to be removed is moved into the recycle folder
   `recycle/<UTC stamp>/<original path>` (tracked files by `git mv`, untracked clutter by plain `mv`, with a
   `recycle/<UTC stamp>/MANIFEST.md` listing each original path and why), never `git rm` or `rm -rf` (owner,
   2026-11-03 14:20 UTC: "can move everything thats gonna be deleted into a recycle folder"); (c) imports, READMEs,
   reproduction scripts and request paths updated in the same commit, and `py_compile` plus a CPU import of every
   slice afterwards. Files pinned by a pack manifest (`submissions/*/…manifest.json`) are moved only after the
   12:00 UTC 4 Nov pack cut-off or as part of a new pack; the packed commits keep reproducing at their URLs either
   way. Historical `outputs/` run records are never rewritten.
   *Owner lines, 2026-11-04 about 00:34 UTC, recorded here because the D2 v5 manifest pins `AGENTS.md` (an edit there breaks
   `d2_poses_v5.sh`): "how about you just fucking do it", read as: phases 3-5 start now, which lifts the 12:00 UTC gate for
   this work and stands as the phase-5 sign-off in `docs/chain/restructure-plan.md`; never-move paths, packed URLs and the
   owner's pushing and uploading are unchanged. Then "your allowed to run opus / sonnet / haiku subagents", which adds Haiku
   to the 2026-09-30 model line in `AGENTS.md`; section 8 is otherwise unchanged.*

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
3. **Work** on the items chosen by section 2, with section 3 in mind, by briefing Codex: route,
   then build and run, then review (section 8). Codex does most of the hands-on work; you lead,
   review and commit. Keep the vertical-slice
   pattern: request, gate, `run.json`, hashed inputs and artifacts. Long GPU work is started
   detached (`setsid nohup ... > outputs/<slice>/<name>.log 2>&1 &`), with its PID and log path
   written down at once.
4. **Measure** every change by running it on real data (usually Codex's job), re-check the headline
   number yourself, and record where the evidence is.
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
- after the freeze only: everything left waits on the owner. Before it, `COMPLETION-PLAN.md` section 3 decides.

Still write the prompt section, so the owner can restart the chain after answering `NEEDS-YOU.md`.

## 8. Collaborating with Codex: Codex does most of the work

The owner asked for Codex collaboration in every context (2026-11-01), then, on 2026-11-02, for
the large majority of the work to be done with Codex. So the Claude context is the **lead**, and
Codex is the **hands**:

- **Claude owns:** reading the state, choosing the items (section 2), writing briefs, running
  the GPU schedule, reviewing every diff, re-checking the numbers that matter, the critic pass,
  commits, the scoreboard and the hand-off.
- **Codex does:** the implementation, the runs (CPU and GPU), the measurement, the analysis,
  figures, documentation and report text, and the fixes the review asks for.

Claude writes code itself only for small fixes (about 20 lines or less) to Codex's work, for
glue nobody else can do, or when Codex is unavailable (see Budget). If you notice you are doing
a stage by hand, stop and brief Codex instead. Codex is still a collaborator, not a rubber stamp,
in both directions: weigh what it says against the evidence, review what it writes, and write
down where you disagree and why.

**One helper, always.** Call Codex only through
`bash scripts/chain/codex_ask.sh <label> <brief.md> [mode] [model] [effort]`. Never hand-roll a
`codex exec` command. The helper adds the safety rules to every brief, keeps the brief, answer and
log under `outputs/chain/codex/`, and logs each call (with its session id) in
`outputs/chain/codex/calls.jsonl`. Launch it with `run_in_background`, end the turn, and read the
answer when you are woken. Never poll it. Write each brief to a file under `.runtime/tmp/` first.

| Mode | Use it for | Sandbox | Default model |
| --- | --- | --- | --- |
| `read-only` | routes, ideas, defect reviews, critic, report reading | read-only | `gpt-6-astra` high |
| `workspace-write` | code, docs, figures, CPU runs and analysis | writes inside the checkout only; **no GPU** (blocked on this WSL machine) | `gpt-6.1-sol` high |
| `gpu` | anything that needs CUDA: teacher depth, training, prediction, scoring on GPU, ONNX checks | **none** (danger-full-access, owner-approved 2026-11-02); one `gpu` job at a time, enforced by `outputs/chain/codex/gpu.lock` | `gpt-6.1-sol` high |

Job modes are capped at 180 minutes (`CODEX_TIMEOUT_MIN` changes it); Codex starts anything
longer detached and reports PID, log and the finish signal. **Follow-ups keep context:**
`CODEX_RESUME=<earlier answer.md> bash scripts/chain/codex_ask.sh ...` sends the new brief into
the same Codex session, so review defects, "now run it on the tuning episodes" and "write the
figure for that" go back to the session that did the work rather than to a cold one.

**Briefs.** A new Codex session starts with no context, so the brief carries the goal, the
measured facts, file paths and line ranges, what has been ruled out, the exact paths it may
write, and a definition of done ("files written, run on real data, numbers and artifact paths
reported"). Every build brief points Codex to section 9. In `gpu` mode nothing enforces paths,
so the brief names them explicitly and says which runtime flags and episodes are in scope.

**The loop for each item:**

1. **Route** (read-only): send the stage, the measured facts, your intended route and the metric.
   Ask for its preferred route, its best unconventional alternative with the cheapest test that
   would separate the two, and the step most likely to fail. New ideas go in `IDEAS.md` with
   "Codex" as the source. You decide the route.
2. **Build and run** (`workspace-write`, then `gpu` or a `CODEX_RESUME` follow-up in `gpu` mode
   when CUDA is needed): Codex implements, runs on real data and reports. Anything with more than a
   couple of trials becomes a loop script (section 9) that Codex builds and you launch and read.
3. **Review** (you, plus a `read-only` Codex defect review of `git diff` in a fresh session for
   behaviour-changing code): read the diff, re-run or re-hash at least the headline number, and
   check the research-integrity rules (section 4) and the build standard (section 9) yourself.
   Send defects back with `CODEX_RESUME`.
4. **Commit** by explicit path (section 6). Codex never commits.

**Parallel work.** Up to three Codex jobs may run at once when their write paths are disjoint;
at most one of them in `gpu` mode, and never while this context or a detached run holds the GPU
(check `.runtime/gpu-stage.lock` and `ps` first). Never give Codex and a Claude sub-agent the same
paths. While jobs run, review the previous job's output or write the next brief; do not redo
Codex's job yourself.

**Every context, at least:** one route call per item, the build and run jobs for each item,
and a Codex **critic** (read-only, fresh session) at the end, beside the Claude critic
(section 2). A cold Codex reading of the extended abstract or a figure, asking what a workshop
reviewer would attack first, is worth one call whenever either changes.

**Budget.** Codex usage is limited but is now the main workforce: plan on roughly 15 to 30 calls
per context, spent on build and run jobs first, and say the running count in your hand-off. If
an answer is empty and the log shows a usage limit, do the work with Claude alone (sub-agents
included) rather than stopping, record it under `### Codex`, and let the next context try again.

## 9. How code is built

The owner's standard (2026-11-02): let code do the heavy lifting (a loop an agent runs and
learns from, not an agent in a loop), do not over-engineer, build clean, keep library code and
implementation clearly apart, use vertical slices, and make it presentable. It binds Claude and Codex alike, and every review checks it. The detailed
removal list is `docs/design/simplification-standard.md`; the slice layout is
`docs/design/feature-template.md`.

- **Code runs the loop; the agent steers it.** Do not put an agent in a loop (run one trial,
  read it, tweak, run the next). Build a loop once that an agent can launch and learn from:
  - Input is a plain file the agent edits: a search space, a queue of configs, or a list of
    ablations, with a trial and time budget.
  - It runs every trial through the slice handlers (gates, `run.json`, GPU lease), skips trials
    already done (identity hash), survives a crash by resuming, and appends one record per trial
    to a ledger (`.jsonl`).
  - It ends by writing one compact summary for the agent: a ranked Markdown table (gates,
    composite, seed spread), the best config, what changed since the last summary, and the
    figures. Then it exits.
  - The agent starts it detached, ends its turn, reads the summary, edits the input and starts
    it again. Agents never hand-run trials one by one or babysit a running loop.
  - Reuse what exists: `track3/features/mesh_generative/` demonstrates a fixed G3 queue,
    source/input checks and resumable admission ledger in `track3/features/mesh_generative/g3.py`.
    The former depth-loop example is preserved under `recycle/20261104T010336Z/`.
    A new stage gets the same shape inside its own slice.
  Within each trial, let existing code do the work: NumPy, SciPy, OpenCV, torch, the `track3/`
  primitives, the official kit and the pinned upstream tools. Hand-rolled maths, parsers,
  loaders or training loops that a library already provides are defects.
- **Do not over-engineer.** Build the smallest thing that does the job and measure it. No
  abstraction before its second real use, no option nobody sets, no registries, plugin systems,
  base classes or wrapper layers, no defensive checks for cases the boundary already rules out.
  Prefer deleting code to adding it.
- **Library code versus implementation.**
  - *Library code* computes: pure functions on arrays and plain values, with no file I/O, no
    paths, no gates, no prints and no run records. It lives in `track3/` (shared, stable
    primitives used by two or more slices) or in a slice's private modules (`_geometry.py`,
    `_model.py`, `_losses.py`), and it can be imported and called on its own.
  - *Implementation* orchestrates: the slice's handler parses the request, calls `authorize`
    and `heavy_execution`, reads inputs, calls library functions, writes artifacts and
    `run.json`. It holds no maths. `__main__.py` and `scripts/` are thin entry points.
  - Move code into `track3/` only when a second slice needs it and its contract is stable.
- **Vertical slices.** A feature owns its request, handler, library modules, runtime flags,
  request drafts and README in `track3/features/<slice>/`. No horizontal service or repository
  layers.
- **Build clean and presentable.** Short functions with clear names and type hints, one
  sentence of docstring, no dead or commented-out code, no history or policy prose in comments.
  Each slice README says what it does, how to run it (one command), what it writes and its
  latest result. Results reach readers as tidy Markdown tables and labelled figures, with
  units and the pseudo-label caveat where it applies.
