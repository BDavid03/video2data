# Needs the owner

Things only the owner can do. Chain contexts add to this list and keep working on everything
else. To grant one, add a dated line to `AGENTS.md` (for example "2026-11-02: push approved").
Every context reads `AGENTS.md` first, so the next context acts on it. Then tick the item here.

## Open

1. [ ] **GitHub remote (D4).** Both packed files must carry a `.../commit/<sha>` URL. The
   checkout became a local git repository on 2026-11-01. Create a GitHub repository (the
   challenge requires Apache-2.0 release for a winning entry), add it as `origin` and push, or
   say the chain may do it. Until then, packed files use the local commit and are marked not
   final.
2. [ ] **Kaggle upload (D6).** When a context has packed the files, it writes the exact
   commands here. Before uploading: export a token from <https://kaggle.com/settings/api> as
   `KAGGLE_API_TOKEN` (a legacy `~/.kaggle/kaggle.json` does not work with kaggle>=2), and
   accept the rules on all six Track 3 competitions (five pose, one CD-O). There is no API for
   accepting rules.

   **Both Track 3 files are ready (context 03, branch `worktree-chain-ctx01`).** Each carries a
   placeholder commit URL. After item 1 (push a commit that contains `7bc1a36` or later), run from
   the checkout that holds the outputs (the worktree `.claude/worktrees/chain-ctx01`, or main after
   copying `outputs/mesh_extents/` and `outputs/pose_insurance/` back), with `<owner>/<repo>` and the
   pushed `<sha>` filled in. The geometry and trajectories are identical to the packed files; only
   the URL column changes.

   ```bash
   K=upstream/v2d_submission_kit/v2d_submission_kit
   URL=https://github.com/<owner>/<repo>/commit/<sha>
   # D1 v2 meshes (CD-O)
   .runtime/kit-env/bin/python $K/tools/pack_reconstruction.py track3-meshes \
     --meshes outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f/meshes \
     --sample $K/data/track_3_mesh_sample_submission.parquet \
     --commit $URL --out submissions/20261101T1605Z-d1-v2/track3_meshes_v2_final.parquet
   # D2 v1 poses (five pose metrics)
   .runtime/kit-env/bin/python $K/tools/pack_submission.py --track track_3 \
     --results outputs/pose_insurance/results-v1 --code-commit $URL \
     --eval-script $K/flash_chord/scripts/evaluate_policy.py \
     -o submissions/20261101T1703Z-d2-v1/track3_poses_v1_final.csv
   .runtime/kit-env/bin/python $K/tools/preflight.py --track track_3 --results outputs/pose_insurance/results-v1
   # upload (needs KAGGLE_API_TOKEN and the rules accepted on all six Track 3 competitions)
   .runtime/kit-env/bin/python $K/submit/v2d_submit.py submissions/20261101T1605Z-d1-v2/track3_meshes_v2_final.parquet \
     --track track_3_mesh -m "Video-measured primitives, vessel+handle rule"
   .runtime/kit-env/bin/python $K/submit/v2d_submit.py submissions/20261101T1703Z-d2-v1/track3_poses_v1_final.csv \
     --track track_3 -m "Static-object insurance rollout"
   ```

   Superseded files (do not upload): D1 v0, v1; D2 v0. Please write the leaderboard scores back
   here or in `AGENTS.md` so the chain can record them in `SUBMISSION.md`.
4. [ ] **Chain contexts cannot edit the main checkout (added by context 01).** Claude Code's
   background-job guard rejects file edits in `/home/ceolwolf/repos/video2data` and requires a
   worktree. Context 01 therefore worked in `.claude/worktrees/chain-ctx01` on branch
   `worktree-chain-ctx01`, and its hand-off is committed there, not in the main checkout the
   driver watches, so the driver will report STALLED. To continue, in the main checkout:
   `git merge --ff-only worktree-chain-ctx01` (main's uncommitted driver edits are untouched; the
   branch does not change AGENTS.md, PROGRAM.md or scripts/chain/), then restart the driver.
   To let later contexts work in place, add `"worktree": {"bgIsolation": "none"}` to
   `.claude/settings.json`; otherwise every context will branch again. Run outputs from context 01
   are under the worktree's `outputs/` (hard-linked copies of the old ones plus the new runs); the
   next context can copy them back with `cp -al`.
   *Update, context 02:* main now holds the context-01 merge, but the guard still refused edits in
   main, so context 02 again worked on `worktree-chain-ctx01` (commits after `3b1520b`). Its
   hand-off is therefore again only on that branch. To continue: in main,
   `git merge worktree-chain-ctx01` (a normal merge; main's extra commits are the owner's merge
   and the `init` gitlink), copy new outputs back
   (`cp -al .claude/worktrees/chain-ctx01/outputs/mesh_extents outputs/` and the same for
   `outputs/mesh_eval`, `outputs/depth/evaluation`, `outputs/depth/public/episode_000039`,
   `outputs/stereo_depth`, `outputs/camera_motion`), and add `"worktree": {"bgIsolation": "none"}`
   to `.claude/settings.json` so the next context can work in place.
   *Update, context 03:* main merged context 02 but `bgIsolation` is still unset, so context 03
   again worked on `worktree-chain-ctx01` (commits `78df274`..the hand-off). To continue: in main,
   `git merge worktree-chain-ctx01`; copy back `outputs/mesh_extents`, `outputs/mesh_eval`,
   `outputs/pose_insurance`, `outputs/report` and `submissions/` with `cp -al`. Also note main's
   merge `9e79099` committed a stray gitlink `.claude/worktrees/chain-ctx01` (mode 160000); it is
   harmless but you may want `git rm --cached .claude/worktrees/chain-ctx01` in main.
   *Notifications (2026-11-02, at the owner's request):* the branch's `scripts/chain/session_chain.py`
   now logs `STRANDED` with the merge command when a newer hand-off exists only in a worktree, and
   rings the terminal bell on that and on every finish (STALLED, LIMIT, COMPLETE). Set
   `CHAIN_NOTIFY_CMD` (a command that takes the message as its last argument, e.g. a Windows toast
   wrapper) for a desktop notification. It takes effect after the merge and a driver restart; the
   running driver has the old code. Contexts also send a PushNotification when their hand-off is
   stranded (rule in the hand-off prompt). You may want a dated `AGENTS.md` line making that rule
   permanent.
5. [ ] **Hand-pose model for D2 (proposal).** Simulated hands must follow a hand reference
   estimated from the egocentric video. The `hand_motion` slice is written for WiLoR (weights)
   plus the MANO hand model; neither is present, and MANO requires registering at
   <https://mano.is.tue.mpg.de> and accepting its licence, which only the owner can do. Approve in
   one line, e.g. "2026-11-0x: WiLoR + MANO approved, MANO files in models/mano/", or say D2 may
   use parked (non-moving) hands, which the chain will try first anyway.
6. [ ] **Docker group (optional).** `docker` is unusable for this user (`/var/run/docker.sock`
   permission denied; sudo needs a password). Not needed now: FlashCHORD runs natively. If the
   official container is ever wanted: `sudo usermod -aG docker ceolwolf`, then restart WSL.
3. [ ] **`coursework/` material** lost on 2026-10-29 (Week 1-12 notebooks, `data/`,
   `glove.6B.100d.txt`, `make_coursework.py`, `helper_funcs/`), if another copy exists. Only the
   hashed subset that `scratch_depth` needs is present. Not needed for the submission.

## Answered

- [x] PDF page renderer (item 7, context 04): approved 2026-11-02 ("approved"). Installed
  without root under `.runtime/tools/poppler/root/` (see `AGENTS.md`); render with
  `LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu .runtime/tools/poppler/root/usr/bin/pdftoppm -r 110 -png <pdf> <prefix>`.
  If you prefer a system install: `sudo apt-get install poppler-utils`.

- [x] Isaac Lab / FlashCHORD install: approved 2026-11-01. Done 2026-11-01 by context 01:
  FlashCHORD at aee95f25 needs no Isaac Sim or Isaac Lab (Newton + MuJoCo-Warp + JAX), so only
  FlashCHORD was installed, natively, in `.runtime/flashchord/venv`. No NVIDIA Isaac Sim licence
  was accepted because Isaac Sim was not installed.
- [x] Kaggle: the owner uploads; the chain packs and preflights only (2026-11-01).
- [x] LaTeX: install a reputable engine from its official source, do not write one
  (2026-11-01).
- [x] Git: local repository initialised 2026-11-01.
- [x] Team name and authors: not important yet; use the anonymous template option
  (2026-11-01).
