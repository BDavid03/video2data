# D2 v5 build plan (Codex plan-d2v5-18, read-only, context 18; execute only after the owner line "2026-11-03: approve D2 v5 heading for episode 19")

D2 v5 remains gated by the exact owner line **“2026-11-03: approve D2 v5 heading for episode 19”** in `AGENTS.md`. H3’s recorded gates pass; approval must arrive by approximately 06:00 UTC November 4, with packing complete by 12:00 UTC.

Two corrections: the actual v4 manifest is `submissions/20261103T0645Z-d2-v4/track3_poses_v4_final.manifest.json`; and v5 retains **19 episodes**—the other 12 members of the 13-episode stereo roster, plus seven outside it.

1. **Build episode 19 from its admitted v4 task.** Read full-precision XYZ+XYZW poses from `outputs/pose_insurance/heading-h3/summary.json:634`. The selected quaternions are approximately:
   - Sweeper: `(-0.0447034104, 0.0164230909, 0.0132305759, 0.9987776725)`.
   - Dust pan: `(0.4715717378, -0.4072180010, 0.5707248370, 0.5348520881)`.

   Preserve positions exactly; convert quaternion ordering to WXYZ for the task. `_placement.py:57`, `apply_conventions`, already implements the counter-rotation mathematics, but chooses convention-derived rotations. Extract its reusable re-expression into a pure helper accepting explicit old/new rotations: `G = Rnewᵀ Rold`, geometry `Gv`, inertia `G I Gᵀ`. Keep the existing convention caller equivalent.

   In `scored_stereo_handler.py`, add the TW build branch: create new mesh/URDF assets, counter-rotate visual geometry and collision-box origin, write the full inertia tensor, and change task rotations/asset paths. Follow `scene.py:146` for asset fields; preserve masses, dimensions, positions, hands and frame count. Record world-geometry and world-inertia equivalence.

   Reuse `scored_scale_handler.inspect:91` against the **TW scene**, not v4. `_scored_scale.physical_checks:20` measures worst first-pose translation/geodesic rotation error: ≤1 cm/≤2°, plus finite/unit quaternions, drift <1 cm/<2°, and export-contract checks. This establishes simulation consistency, not scored accuracy.

2. **Fold in the four engineering fixes.** Paths below are within `track3/features/pose_insurance/`.

   | Fix | Files touched |
   |---|---|
   | Dedicated `scored_stereo_authorized` flag; explicit owner/H3 admission for TW | `runtime.json`, `scored_stereo_handler.py`, `scored_stereo_loop.py` |
   | NumPy and numerical dependency imports only after runtime/episode admission and `heavy_execution`; cover transitive imports too | `scored_stereo_handler.py`, `scored_stereo_loop.py`, `scored_stereo_pack.py` |
   | Move protocol-specific `analysis` orchestration out of the numerical module | `_stereo_placement.py`, `stereo_placement_report.py` |
   | Extract Markdown rendering from `scored_stereo_loop.py:262` | `scored_stereo_loop.py`, new `scored_stereo_report.py` |

   The v4 manifest directly pins all existing Python files above and `_placement.py` (`:310`), plus `scored_stereo_queue.json` (`:214`) and the v4 reproduction script (`:358`). Runtime configuration is pinned indirectly through build/run inputs. Preserve historical artifacts and hashes; changes belong to the new committed pack. Leave the v4 queue/script unchanged.

   Add `scored_heading_v5.py`, `scored_heading_v5_queue.json`, and README instructions: one frozen episode-19 plan, one attempt, resumable ledger, and composition from v4. Copy the other 19 Parquets verbatim; compare SHA-256 against v4. Compare all **8,650 non-19 CSV rows** field-for-field, including signed-zero text, excluding only `code_commit_url`. Whole-row byte identity is impossible because that URL must change globally.

3. **Run one new policy, episode 19 only.** Preserve 512 steps, one world, replay minimum 32, official evaluator and `GpuLease`. Existing evidence: `stereo-placement-scored/ledger.jsonl:38` records **33.388 seconds**; the first cold v4 run took **178.088 seconds** (`:24`). Budget 3–5 minutes.

   Planned command, after implementing the new entry point, inside one `gpu`-mode job:

   ```bash
   PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .runtime/linux-cuda/bin/python -B \
     -m track3.features.pose_insurance.scored_heading_v5 \
     --queue track3/features/pose_insurance/scored_heading_v5_queue.json --phase all
   ```

   Freeze sources before execution. Failure retains v4; no basin switching or extra trials.

4. **Commit, pack, document.** Claude reviews and commits the implementation, configuration and new `scripts/reproduce/d2_poses_v5.sh`. Its full SHA—not `d1d1da5`—becomes `https://github.com/BDavid03/video2data/commit/<SHA>`; the owner pushes it.

   Extend `scored_stereo_pack.py` for v5 admission and CPU reproduction. The new shell wrapper invokes that version. Set `PACK=submissions/<actual-UTC-stamp>-d2-v5` and `URL` to that committed URL:

   ```bash
   bash scripts/reproduce/d2_poses_v5.sh --run --code-commit "$URL" \
     --out "$PACK/track3_poses_v5_final.csv"
   ```

   Manifest: existing fields plus hashed v4 parent, approval, H3 evidence, TW plan, new build/run/checkpoint, 20 export hashes, composition, physical/invariance evidence, committed sources, reproduction script, commands, CSV hash and preflight verdict.

   From kit root, with `W` the absolute repository root:

   ```bash
   "$W/.runtime/kit-env/bin/python" tools/preflight.py --track track_3 \
     --results "$W/outputs/pose_insurance/results-v5"
   ```

   Record a separate `--skip-kaggle` result if credentials block full preflight, as the [kit skill](upstream/v2d_submission_kit/v2d_submission_kit/skills/v2d-submit/SKILL.md) distinguishes.

   NEEDS-YOU’s owner-only command:

   ```bash
   "$W/.runtime/kit-env/bin/python" submit/v2d_submit.py \
     "$W/$PACK/track3_poses_v5_final.csv" --track track_3 \
     -m "Static stereo placement; episode 19 H3 TW headings"
   ```

   SUBMISSION row: actual UTC, file, SHA-256, 9,120 rows, full commit URL, preflight status, “episode 19 only; 19 exports identical”, not uploaded.

5. **Estimate:** 2–3 hours including implementation, equivalence checks, review and packing. Highest implementation risk: URDF/inertia frame conversion. Scientific risk remains the near-tied sweeper basin; G2 passed by only **0.00077 AUC**.

Files changed: none. Commands run: read-only `pwd`, `rg`, `ls`, `nl`, `sed`, `head`, `tail`. Results: source/artifact inspection only; no checks executed or artifacts created. Left undone: all implementation, GPU execution, verification, committing and packing, pending owner approval.