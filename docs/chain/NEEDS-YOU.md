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

   **Both Track 3 files are ready (D1 v4 from context 08, D2 v3 from context 05; branch `worktree-chain-ctx01`).** Each carries a
   placeholder commit URL. After item 1 (push a commit that contains context 08's last commit or later), run from
   the checkout that holds the outputs (the worktree `.claude/worktrees/chain-ctx01`, or main after
   copying `outputs/mesh_extents/`, `outputs/pose_insurance/` and `outputs/stereo_depth/` back), with `<owner>/<repo>` and the
   pushed `<sha>` filled in. The geometry and trajectories are identical to the packed files; only
   the URL column changes.

   ```bash
   K=upstream/v2d_submission_kit/v2d_submission_kit
   URL=https://github.com/<owner>/<repo>/commit/<sha>
   # D1 v4 meshes (CD-O; rack compartment + thin-tool profiles, exact local CD-O 0.6629 cm pair-weighted vs v3 0.8115)
   .runtime/kit-env/bin/python $K/tools/pack_reconstruction.py track3-meshes \
     --meshes outputs/mesh_extents/build/20261102T051416.149421Z-f377e0a1/meshes \
     --sample $K/data/track_3_mesh_sample_submission.parquet \
     --commit $URL --out submissions/20261102T0516Z-d1-v4/track3_meshes_v4_final.parquet
   # (or rebuild end to end: bash scripts/reproduce/d1_meshes.sh $URL --version v4 --run)
   # D2 v3 poses (five pose metrics)
   .runtime/kit-env/bin/python $K/tools/pack_submission.py --track track_3 \
     --results outputs/pose_insurance/results-v3 --code-commit $URL \
     --eval-script $K/flash_chord/scripts/evaluate_policy.py \
     --skip-crosscheck --out submissions/20261102T0032Z-d2-v3/track3_poses_v3_final.csv
   .runtime/kit-env/bin/python $K/tools/preflight.py --track track_3 --results outputs/pose_insurance/results-v3
   # upload (needs KAGGLE_API_TOKEN and the rules accepted on all six Track 3 competitions)
   .runtime/kit-env/bin/python $K/submit/v2d_submit.py submissions/20261102T0516Z-d1-v4/track3_meshes_v4_final.parquet \
     --track track_3_mesh -m "Video-measured primitives: hollow vessels, compartment rack, thin-tool profiles"
   .runtime/kit-env/bin/python $K/submit/v2d_submit.py submissions/20261102T0032Z-d2-v3/track3_poses_v3_final.csv \
     --track track_3 -m "Static objects, stereo-scaled placement"
   ```

   **Spoon risk in D1 v4 (please read before uploading).** v4 changes the wooden spoon, which
   has 10 of the 36 scored pairs and no public scan, from a box to a bowl-and-handle shape. The
   pre-registered rule required this (the same thin-tool rule passed on the scanned dust pan and
   sweeper), but nothing measured the spoon itself, and its depth ratios are medium confidence.
   The new spoon differs from the old box by 1.32-1.39 cm CD-O. If it were on average 0.39 cm
   worse than the box against the host's spoon, it would cancel the whole measured gain over the
   36 pairs. The chain recommends v4 as packed and did not pack a spoon-as-box variant: uploading
   both and keeping the better score would choose a shape on the host's references.
   *Update, context 09:* a scan-free check fixed before scoring (silhouette agreement with the
   opening video frame) could not be validated on the scanned objects (it ranked shapes like
   CD-O on only 5 of 8 objects), so it cannot score the spoon. Descriptively, the v4 spoon's
   outline matches the video silhouette much better than the old box (mean IoU 0.72 vs 0.49 over three views), which
   is reassuring about its outline but says nothing certain about its CD-O. The recommendation
   is unchanged (`outputs/mesh_extents/spoon-proxy/summary.md`).

   Superseded files (do not upload): D1 v0, v1, v2, v3; D2 v0, v1, v2. Please write the leaderboard scores back
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
   *Update, context 04:* main again equalled the branch at start (`aa3b295`), but the guard still
   refused edits in main, so context 04 worked on `worktree-chain-ctx01` (commits after `aa3b295`).
   To continue: in main, `git merge worktree-chain-ctx01`; copy back with `cp -al`
   `outputs/pose_insurance`, `outputs/depth/public/episode_000002`, `outputs/depth/public/episode_000027`,
   `outputs/mesh_extents/public/episode_000002`, `outputs/mesh_extents/public/episode_000027`, and
   `submissions/20261101T2315Z-d2-v2`. `.runtime/tools/poppler` already lives in main's `.runtime`.
   *Update, context 05:* main again equalled the branch at start (`e24414f`) and the guard still
   refused edits, so context 05 worked on `worktree-chain-ctx01` (commits after `e24414f`). To
   continue: in main, `git merge worktree-chain-ctx01`; copy back with `cp -al`
   `outputs/pose_insurance` (new `scale/`, `v3/`, `results-v3/` and run dirs), `outputs/stereo_depth`,
   `outputs/mesh_extents/build/20261102T000114.850697Z-1a4c354e` and
   `submissions/20261102T0032Z-d2-v3`.
   *Update, context 06:* main again equalled the branch at start (`5972bdd`) and the guard still
   refused edits, so context 06 worked on `worktree-chain-ctx01` (commits after `5972bdd`). To
   continue: in main, `git merge worktree-chain-ctx01`; copy back with `cp -al`
   `outputs/pose_insurance/shrink` and `outputs/mesh_eval/pairs`. No packed file changed.
   *Update, context 07:* main equalled the branch at start (`be0aab8`); the guard still refused
   edits, so context 07 worked on `worktree-chain-ctx01` (commits after `be0aab8`). In main,
   `git merge worktree-chain-ctx01`; copy back with `cp -al` `outputs/mesh_eval`,
   `outputs/mesh_extents/build` and `submissions/20261102T0313Z-d1-v3` (the new D1 v3 pack).
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
