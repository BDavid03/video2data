"""Present scored S1 estimates, physical outcomes and hybrid packaging evidence."""
from pathlib import Path
from track3.provenance import write_json
from track3.runs import ROOT


def report(plans: dict, decisions: dict, package: dict) -> None:
    """Write a per-episode account with explicit no-op and fallback reasons."""
    output = ROOT / 'outputs/pose_insurance/v3'
    lines = ['# D2 v3: stereo-anchored offset scale', '',
             f"Package status: **{package['status']}**.", '',
             'S1 was adopted by the committed decision in `docs/chain/pose-scale-protocol.md`. '
             'Each changed task starts from its exact D2 v2 producer and changes only slot 1’s table-plane offset. '
             'The budget stays 512 environment steps, one world, replay minimum 32. '
             'Each changed episode receives one serial training/evaluator attempt; failures keep v2.', '',
             '| Episode | s | Observed frames | Median support | Median log-ratio MAD | Valid | Scene changed | First pose cm / deg | Drift cm / deg | Selected export / reason |',
             '|---:|---:|---:|---:|---:|:---:|:---:|---|---|---|']
    fmt = lambda v: '—' if v is None else f'{v:.6f}'
    for ep in sorted(plans):
        p, d = plans[ep], decisions[ep]
        s, c = p['stereo'], d.get('checks', {})
        first = f"{fmt(c.get('first_pose_cm'))} / {fmt(c.get('first_pose_deg'))}"
        drift = f"{fmt(c.get('drift_cm'))} / {fmt(c.get('drift_deg'))}"
        selected = 'S1' if d['accepted'] else 'v2'
        lines.append(f"| {ep} | {fmt(s['s']) if s else '—'} | {s['observed_frames'] if s else '—'} | "
                     f"{fmt(s['median_support_share']) if s else '—'} | {fmt(s['median_log_ratio_mad']) if s else '—'} | "
                     f"{s['valid'] if s else 'n/a'} | {p['changed']} | {first} | {drift} | {selected}: {d['reason']} |")
    lines += ['', 'Unchanged single-object episodes have no new physical run. Their drift is the previously recorded v2 value; '
              'a dash means the historical record did not report that first-pose quantity.', '',
              '## Composition', '', '| Episode | Source | SHA-256 |', '|---:|---|---|']
    for e in package.get('exports', []):
        ep = int(Path(e['path']).stem.split('_')[1])
        lines.append(f"| {ep} | `{e['source']}` | `{e['sha256']}` |")
    if package['status'] == 'packed':
        lines += ['', f"Packed file: `{package['file']}`", '', f"SHA-256: `{package['sha256']}`", '',
                  f"Rows: **{package['rows']}**. Kit preflight: **{package['preflight_verdict']}**.", '',
                  f"Source/artifact verification: `{package['verification_summary']}`.", '',
                  f"CPU package reproduction: **{package['reproduction']['identical']}**; "
                  f"fresh SHA-256 `{package['reproduction']['actual_sha256']}`.", '',
                  f"Reproduction evidence: `{package['reproduction']['manifest']}`.", '',
                  f"Manifest: `{package['manifest']}`. Run record, pack/preflight logs and verify output are beside it."]
    else:
        lines += ['', f"No package: {package['reason']}"]
    lines += ['', '## Caveats and provenance', '',
              '- Evaluation videos ship no stereo calibration. The unchanged stereo slice uses the public C-left/B-right '
              'calibration (75.47 mm baseline), assumed to hold for the evaluation rig.',
              '- The scale idea came from train mocap distance ratios. Selection used train episodes because tuning 39/42 '
              'have one object and cannot measure placement, as disclosed by the fixed protocol.',
              '- S1 preserves heights, rotations, meshes, dimensions, collisions, inertia and mass; only the planar offset changes. '
              'It is applied after the incumbent’s placement/clearance, with no new separation or adjustment.',
              '- Part of the error reduction may be a hedge from shortening offsets with wrong directions. '
              'Train distance ratios remain 1.18–2.23 and direction errors 2–130 degrees.',
              '- No host-score gain is claimed. AUC/SP-SR are expected to remain effectively unchanged; '
              'the recorded train AUC moved only 0.268844 to 0.269024, with SP-SR and MP-SR zero.',
              '- The code URL is a nonfinal owner placeholder at bb8a3f0. Working-source hashes accompany the package; '
              'the lead reviews and commits the implementation. No uploads, installs, labels or holdouts enter this stage.',
              '- Stereo runtime authorization added exactly the sixteen named evaluation episodes and enabled '
              '`evaluation_authorized`; public authorization was retained. Pose runtime added only `scored_scale_authorized`.',
              '', 'Per-episode stereo frame identities and anchors are in `stereo_<episode>.json`; plans and evaluator '
              'attempts are pinned by `plans.json`, `decisions.json` and append-only `ledger.jsonl`.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    write_json(output / 'report.json', package)
