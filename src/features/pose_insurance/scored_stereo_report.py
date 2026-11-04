"""Render the scored stereo stage without numerical dependencies or file I/O."""


def markdown(plans: dict, decisions: dict, manifest: dict, executed: int, resumed: int,
             preflight: dict, episodes: tuple) -> str:
    """Render physical evidence, frozen decisions and preflight status."""
    lines = ['# T1 scored-stage physical result', '',
             f"New GPU runs: {executed}; resumed attempts: {resumed}. Accepted T1: {len(manifest['composition']['T1'])}/20.", '',
             'No reference poses or scan geometry entered this stage. Public stereo calibration is assumed '
             'to hold for the evaluation rig. The train idea and gates were informed by train diagnostics; '
             'fresh confirmation failed by the letter and the owner approved the candidate anyway. '
             'The expected gain is RPE/MPPE; direction remains unobservable. No final file has been packed.', '',
             '| Episode | Result / reason | Stereo / S1 length (cm) | c | First pose (cm / deg) | Drift (cm / deg) | Export |',
             '|---:|---|---:|---:|---:|---:|---|']
    for ep, plan in sorted(plans.items()):
        decision = decisions[ep]
        checks = decision.get('checks', {})
        lengths = '—' if plan['stereo_length_m'] is None else f"{100*plan['stereo_length_m']:.3f} / {100*plan['s1_length_m']:.3f}"
        if plan['stereo_length_m'] is None and plan['s1_length_m'] is not None:
            lengths = f"— / {100*plan['s1_length_m']:.3f}"
        factor = '—' if plan['clearance']['factor'] is None else f"{plan['clearance']['factor']:.9f}"
        first = '—' if 'first_pose_cm' not in checks else f"{checks['first_pose_cm']:.6f} / {checks['first_pose_deg']:.6f}"
        drift = '—' if 'drift_cm' not in checks else f"{checks['drift_cm']:.6f} / {checks['drift_deg']:.6f}"
        result = 'T1' if decision['accepted'] else 'v3 fallback' if ep in episodes else 'v3 unchanged'
        lines.append(f"| {ep} | {result}: {decision['reason']} | {lengths} | {factor} | {first} | {drift} | `results-v4/episode_{ep:06d}.parquet` |")
    lines += ['', 'Physical limits: first pose ≤1 cm / ≤2 degrees; drift <1 cm / <2 degrees. '
              'Unchanged/fallback exports are rechecked on CPU against their admitted v3 task.', '',
              f"Full preflight: exit {preflight['exit_code']}, ready={preflight['ready']} (`preflight.log`). "
              f"Local preflight with Kaggle checks skipped: {preflight.get('local', 'not needed')} (`preflight-local.log`).", '',
              'Frozen clearance factors and projected gaps: `plans/episode_*.json`. '
              'GPU attempts and optimizer/physical evidence: `data/runs/*/episode_*/run.json`. '
              'Export hashes, source policies and composition: `composition.json`. '
              'The lead must commit before packing; `scripts/reproduce/d2_poses_v4.sh` verifies and reproduces v4 on CPU.', '']
    return '\n'.join(lines)
