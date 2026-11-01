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
