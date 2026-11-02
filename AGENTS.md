# Current user instructions

- On 2026-11-03 (local; 2026-11-02 22:30 UTC, chain context 12) the user wrote, live in the session:
  "needs-you now approved". Read as approval of `docs/chain/NEEDS-YOU.md` item 7, the only open
  decision the chain itself can act on: build the stereo-placement pose candidate (slot-1 offset from
  direct calibrated-stereo centroids) despite the unmeasurable episode 13, judged on the train
  episodes under a new protocol before any policy run or repack. It does not change the 2026-11-03
  line below: the user still pushes and uploads personally; MANO registration and the docker group
  still need the user's own logins.
- On 2026-11-03 the user wrote "all owner requirements approved", then: "i said id do the kaggle
  manually just get creative with the model and data / data manipulation". So:
  - The user pushes to GitHub and uploads to Kaggle personally. The chain never pushes or uploads; it
    keeps the packed files and the exact commands in `docs/chain/NEEDS-YOU.md` current. The real-URL
    D1 v4 / D2 v3 files were packed at commit `15060c1` (with an Apache-2.0 `LICENSE`).
  - The chain's work is now creative improvement of the models and of the data and its manipulation
    (new representations, using more of each clip, fusing existing depth and masks, augmenting or
    re-deriving data), within `docs/chain/PROGRAM.md` sections 3, 4 and 9: protocol before scoring,
    choices on tuning data, episodes 41/43/44 excluded, no tuning against the leaderboard. A better
    method found this way may be packed as a new version for the user to upload.
- On 2026-11-02 (chain context 04) the user replied "approved" right after the context listed
  its new proposal to install poppler-utils for PDF page rendering (`docs/chain/NEEDS-YOU.md`
  item 7). Read as approval of that install. `sudo` needs a password, so the context installed
  Ubuntu's own packages without root: `apt-get download` of poppler-utils and libpoppler156
  26.01.0-2ubuntu0.3 plus their missing libraries, unpacked with `dpkg-deb -x` into
  `.runtime/tools/poppler/root/`, run with `LD_LIBRARY_PATH` set to its `usr/lib/x86_64-linux-gnu`.
- Later on 2026-11-02 the user added: "also in instructions let \"code\" do the heavy lifting,
  dont over-engineer, build clean, clear seperation between library code & implementation,
  vertical slice, presentable format", then explained the first point: "i meant rather than try
  and agent in a loop, make a loop that an agent can run and feed off the results". This is
  `docs/chain/PROGRAM.md` section 9, repeated in the preamble of every
  `scripts/chain/codex_ask.sh` call: repeated trials run in a resumable loop script that writes a
  ledger and a summary the agent reads and steers from (never an agent hand-running trials),
  existing libraries do the work inside each trial, nothing is over-engineered, pure library code is kept apart from slice handlers,
  features stay in vertical slices, and code, READMEs and results are presentable.
- On 2026-11-02 the user asked, for `scripts/chain/session_chain.py`: "do large majority of
  work with codex". Each chain context now leads (chooses items, briefs, reviews diffs, re-checks
  headline numbers, commits, hands off) and Codex does most of the hands-on work: code, runs,
  analysis, figures and text (`docs/chain/PROGRAM.md` section 8). Codex's workspace-write sandbox
  blocks the GPU on this WSL machine, and the user chose to allow unsandboxed Codex jobs for GPU
  work: `scripts/chain/codex_ask.sh` mode `gpu` runs Codex with danger-full-access, one such job
  at a time. Codex still never commits, installs, uploads or deletes outside `.runtime/tmp/`.
- Later on 2026-11-01 the user asked: "ensure codex collaboration please". Every chain context
  now works with Codex as `docs/chain/PROGRAM.md` section 8 describes, through
  `scripts/chain/codex_ask.sh` only: route and idea consultation and a cross-family critic in
  every context, plus diff reviews and bounded workspace-write jobs where they help. Letting
  Codex write files (exclusive paths, reviewed and committed by the Claude context) is my
  reading of "collaboration"; Codex never commits, installs or uploads.
- On 2026-11-01 the user answered the chain's owner list (`docs/chain/NEEDS-YOU.md`) in these
  words: "Isaac Lab/FlashCHORD is approved for install, ill upload to kaggle, dont make a latex
  engine, find a reputable one online, init a git repo, team name is and author not important
  right now, i want you to focus on taken the road of less travel to extract the best results".
  So:
  - Installing Isaac Lab and the FlashCHORD runtime (and the packages they pull in, `hydra`
    among them) is approved, for the Track 3 pose submission.
  - Installing a reputable, established LaTeX engine from its official source is approved
    (for example the Tectonic release binary or TeX Live). Do not write an engine.
  - This checkout is now a git repository (initialised 2026-11-01). Commit finished work by
    explicit path. Creating or pushing a GitHub remote was not mentioned and stays the user's.
  - The user uploads to Kaggle. The chain stops at packed files with a clean kit preflight,
    and never uploads or accepts competition rules.
  - Team name and authors do not matter yet: use the template's anonymous option.
  - Strategy: take the road less travelled to get the best results (see
    `docs/chain/PROGRAM.md` section 3).
  The chain driver and its brief are `scripts/chain/session_chain.py` and `docs/chain/`. Each
  context there follows `docs/chain/PROGRAM.md`.
- On 2026-10-29 the user moved the working and artifact root to this WSL
  checkout (`/home/ceolwolf/repos/video2data`) and asked for a Linux CUDA
  environment: "use here", "install linux cuda dont use the windows mount".
  `.runtime/linux-cuda` (uv-managed Python 3.13, torch 2.7.1+cu128) is the
  interpreter; `configs/project.json: artifact_roots` lists permitted roots.
  Do not read or write `/mnt/z`. Runs are authorised ("continue, runs are
  authorised"): the continuous Codex-consulted loop in
  `track3/features/auto_research/` executes experiments, records, visualises
  and improves them. Episode 41 stays excluded; holdout 43/44 only via finalize.
- Later on 2026-10-29 the user raised the bar: selection must not rest on MAE
  alone ("if mae is low but the model is shit it cant be used"), the winner is
  saved as ONNX, and the study is "PhD level, across the board". The loop now
  selects by usability gates plus a composite (log-MAE, SILog, delta1, bias,
  edge fidelity); `track3/features/depth_student/` is the main model line
  (UNet/dilated/coursework, SILog/gradient losses, EMA, augmentation, up to
  512 px, replication across seeds, ONNX export). onnx/onnxruntime/onnxscript
  were added to the Linux environment for that export. The user also wants
  over-engineered code simplified (example: `track3/calibration/geometry.py`,
  rewritten with NumPy).

- On 2026-10-18 the user expanded execution authorization: "all flags are go,
  every experiment ran, go deep, follow your gut." Proceed with bounded research
  experiments, original scratch CNN training/inference, ensembles, optimization,
  existing-toolchain compiler experiments and a frozen held-out assessment.
  This supersedes the earlier research execution and holdout restrictions below.
  Episode 41 remains excluded, storage stays on Z:, external environments remain
  read-only, and new production dependencies still need confirmation. Freeze
  model/ensemble choices on tuning evidence before accessing holdout outcomes.
- The expanded 2026-10-18 campaign completed its frozen holdout assessment.
  Those outcomes must not guide later recipe changes or synthetic mixing ratios.
  The user's synthetic-enrichment request also produced a separate 128-sample
  procedural training-only pilot; no synthetic samples entered that campaign.
  Keep synthetic provenance explicit and select future mixtures on real tuning
  data. A fresh unbiased assessment needs an independently fixed protocol.
- On 2026-10-18 the user permitted an execution run. This supersedes the
  implementation-only restriction for the selected run; it does not authorize
  tests, dependency changes, notebooks, simulation or submission uploads.
  The user selected the default 12-trial tuning run, including MoGe2 targets and
  preparation. The selected budget uses the ten training/tuning episodes only,
  20 teacher frames per episode, stride-five preparation at maximum side 128,
  and the default bounded search. Episode 41 and final holdout remain excluded.
- Absolutely no testing. Do not run test suites, smoke checks, validation commands,
  or further dependency-resolution experiments unless the user changes this instruction.
- Continue implementation and documentation without running those checks.
- Use vertical slice architecture for new work. Keep each feature's request,
  handler, integrations, persistence, configuration and documentation together.
  Share stable technical primitives; do not add horizontal service/repository layers.
- The current depth-stage scope is implementation only. The user explicitly chose
  not to run it on episode 41. Keep model execution disabled.
- The user also requested a depth model from scratch using `coursework/`.
  Its vertical slice is `track3/features/scratch_depth/`. Preparation, training and
  inference remain disabled; the request does not authorize executing notebooks.
- Keep all generated artifacts, environments, caches and temporary outputs inside
  `Z:/hugging-face/video2data`.
- Existing external environments may be reused read-only; do not modify them.
- Ask for confirmation before adding new production dependencies.
- The user approved the MoGe2-only proposal on 2026-09-25: the 69 locked wheel
  packages, three pinned source trees and single model file described in
  `configs/perception/moge/README.md`. That installation needs no repeat approval.
  It does not authorize tests, model execution, simulation or additional packages.
- Prefer pnpm if dependency installation for JavaScript is later authorized.
- Sub-agents run on Opus 5.5 or Sonnet 5.5. The user set this on 2026-09-30. It
  replaces the agent model settings in `TRACK3-PLAN.md`.
- On 2026-10-11 the user approved available GPT sub-agents for this buildout
  because Opus/Sonnet sub-agents were unavailable in the session. This is a
  scoped exception to the preceding model preference.
- The 2026-10-11 buildout request includes PyTorch optimization, custom CUDA
  kernels, ensembles, ANN/CNN/transformer models, gradient boosting and bounded
  hyperparameter planning. Implement these with execution disabled. The user
  requested bare-minimum testing; this pass runs no checks and does not grant
  preparation, training, inference, notebook or compiler execution permission.
- On 2026-09-30 the user asked to continue the project design documents after the
  organisers opened submissions. The Submission Kit and the FlashCHORD source at
  `aee95f25` were downloaded into `upstream/` as read-only references. Neither was
  executed or installed, and that download authorizes neither. The design set is
  in `docs/design/` and in each feature's README.

The no-testing direction supersedes the earlier agreement to run npm test.
Existing test files and historical results predate this direction and are retained.
