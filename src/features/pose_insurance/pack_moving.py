"""Compose gated moving rollouts with frozen D2 v5 exports, entirely on CPU."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
from src.paths import current_name
from src.provenance import sha256_file
from ._moving import tracking_metrics

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / 'upstream/v2d_submission_kit/v2d_submission_kit'
V5_MANIFEST = 'submission/20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json'
V5_HASH = '0b2606aed015fe0a64e31037ff790b5483680c3da7e7a5ae2d503edeb8261035'
V5_MANIFEST_HASH = '0fcbfd7d48dc137fc5598adbcb8d609c5de3dd717a87ceb886b9b885e7933f7d'
SCRATCH = ROOT / 'data/.runtime/tmp/mpack-dry'
OUTPUT = ROOT / 'data/outputs/pose_insurance/moving-pack'
POSE = [f'{kind}_{axis}' for kind, axes in (('position', 'xyz'), ('quaternion', 'xyzw')) for axis in axes]
M1_LIMITS = {
    'camera_heldout_reprojection_median_px640': ('le', 1/3),
    'camera_heldout_reprojection_p95_px640': ('le', 1.),
    'camera_heldout_world_residual_median_m': ('le', .01),
    'camera_heldout_tracks_min': ('ge', 30), 'camera_heldout_regions_min': ('ge', 4),
    'camera_coverage_fraction': ('ge', .98), 'table_residual_rms_m': ('le', .005),
    'table_disjoint_patch_normal_max_deg': ('le', 2.), 'mask_usable_fraction': ('ge', .9),
    'mask_longest_unusable_interval_frames': ('le', 2),
    'object_30_stereo_correspondence_fraction': ('ge', .9),
    'object_fb_translation_median_m': ('le', .01), 'object_fb_rotation_median_deg': ('le', 5.),
    'object_speed_max_m_s': ('le', 1.5), 'object_angular_rate_max_deg_s': ('le', 540.),
    'object_reference_valid_fraction': ('ge', .9),
    'reprojection_iou_median': ('ge', .6), 'reprojection_iou_p10': ('ge', .3),
    'reprojection_coverage_fraction': ('ge', .9), 'frame0_base_gap_m': ('le', .005),
    'frame0_penetration_m': ('le', .005), 'zero_action_support_translation_m': ('le', .01),
    'zero_action_support_rotation_deg': ('le', 2.),
}


def path(value: str | Path) -> Path:
    """Resolve a checkout-owned input without accepting excluded episode paths."""
    name = str(value)
    if name.startswith('/mnt/') or '..' in Path(name).parts:
        raise ValueError('External or parent-relative input path')
    if Path(name).is_absolute():
        name = str(Path(name).relative_to(ROOT))
    result = (ROOT / current_name(name)).resolve()
    result.relative_to(ROOT)
    if any(int(ep) in {2, 13, 22, 24, 27, 41, 43, 44}
           for ep in re.findall(r'(?:episode[_=-]?|ep)(\d+)(?=\D|$)', str(result))):
        raise ValueError('Excluded episode input path')
    return result


def entry(value: str | Path) -> dict:
    """Record a local input's exact current content identity."""
    p = path(value)
    return dict(path=str(p.relative_to(ROOT)), sha256=sha256_file(p))


def read(value: str | Path) -> dict:
    return json.loads(path(value).read_text())


def checked(value: dict) -> Path:
    p = path(value['path'])
    if sha256_file(p) != value['sha256']:
        raise ValueError(f'Input hash mismatch: {p}')
    return p


def write(value: Path, data: dict) -> None:
    value.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Missing or nonfinite measurement: {label}')
    return float(value)


def require_reprojection(data: dict, label: str) -> None:
    for key, minimum in (('median_iou', .60), ('p10_iou', .30), ('usable_fraction', .90)):
        value = number(data.get(key), f'{label}.{key}')
        if not minimum <= value <= 1.:
            raise ValueError(f'{label} reprojection fails {key}')


def require_tracking(data: dict, label: str) -> None:
    t, q, e, es, theta, ts = [number(data.get(k), f'{label}.{k}')
                             for k in ('T', 'Q', 'E_m', 'E_S_m', 'Theta_deg', 'Theta_S_deg')]
    if t < .5 or q < .5 or min(e, es, theta, ts) < 0 or e > min(.5 * es, .03) or theta > min(.5 * ts, 15.):
        raise ValueError(f'{label} fails T/Q/E/Theta adoption bounds')


def require_m1(data: dict, episode: int) -> None:
    if data.get('episode') != episode or data.get('label_free') is not True or data.get('passed') is not True:
        raise ValueError('Clause 2 requires this episode\'s passed label-free M1 evidence')
    for key, (direction, limit) in M1_LIMITS.items():
        row = data.get('gates', {}).get(key, {})
        value = number(row.get('value'), f'M1.{key}')
        fraction_invalid = ('fraction' in key or 'iou' in key) and value > 1.
        if value < 0 or fraction_invalid or (value > limit if direction == 'le' else value < limit) or row.get('passed') is not True:
            raise ValueError(f'Clause 2 M1 fails {key}')


def require_m2v(data: dict) -> None:
    """Require the sealed M2v GO, including achieved-pose video reprojection."""
    if data.get('passed') is not True or data.get('verdict') != 'GO' or data.get('episode') != 39:
        raise ValueError('Clause 1 requires passed M2v GO evidence on 39')
    require_tracking(data['tracking'], 'M2v')
    if not .45 <= number(data.get('AUC'), 'M2v.AUC') <= 1.:
        raise ValueError('Clause 1 M2v AUC outside [.45, 1]')
    require_reprojection(data['reprojection'], 'M2v')


def require_completion(official: dict, checkpoint_hash: str, count: int) -> None:
    """Require a full explicit-reset evaluator attempt with no termination."""
    tracking = official.get('reference_tracking', {})
    failures = tracking.get('failure_cause_fraction', {})
    if (official.get('checkpoint_sha256') != checkpoint_hash or official.get('reset_mode') != 'explicit'
            or official.get('non_finite_worlds') != [] or official.get('start_frame') != 0
            or official.get('world_count') != 1 or official.get('step_count') != count
            or tracking.get('reference_end_fraction') != 1. or not {'object', 'wrist'} <= failures.keys()
            or any(number(v, 'failure fraction') != 0 for v in failures.values())
            or number(official.get('environment_steps'), 'environment_steps') < 1):
        raise ValueError('Clause 3 requires the end reached without a reset or termination')


def require_first(row: dict) -> None:
    if row['first_position_m'] > .01 or row['first_orientation_deg'] > 2.:
        raise ValueError('Clause 4 first pose exceeds 1 cm / 2 degrees')


def official_roster() -> tuple[dict, dict]:
    """Load only the public scored row roster and official object slot order."""
    sys.path.insert(0, str(KIT))
    from tools.pack_submission import public_track3_grid
    config = read(KIT / 'config/tracks.json')['tracks']['track_3']
    episodes = config['scored_episodes']
    if set(episodes) & {2, 13, 22, 24, 27, 39, 41, 42, 43, 44}:
        raise ValueError('Unexpected scored episode roster')
    return public_track3_grid(KIT / 'data/track_3_sample_submission.csv',
                             KIT / 'data/track_3_evaluation_objects.json', episodes)


def frozen_v5() -> dict:
    """Verify the frozen v5 pack and all twenty export byte hashes."""
    if sha256_file(path(V5_MANIFEST)) != V5_MANIFEST_HASH:
        raise ValueError('Frozen D2 v5 manifest identity mismatch')
    manifest = read(V5_MANIFEST)
    if manifest['version'] != 'v5' or manifest['sha256'] != V5_HASH or sha256_file(path(manifest['file'])) != V5_HASH:
        raise ValueError('Frozen D2 v5 CSV identity mismatch')
    roster, _ = official_roster()
    exports = manifest['exports']
    if len(exports) != len(roster) or {int(Path(e['path']).stem.split('_')[1]) for e in exports} != set(roster):
        raise ValueError('Frozen v5 exports do not cover the scored roster exactly')
    for export in exports:
        checked(export)
    return manifest


def evaluator_inputs() -> tuple[dict, str]:
    """Require the kit's unchanged script and pinned upstream evaluation modules."""
    kit_manifest = read(KIT / 'KIT_MANIFEST.json')
    for name in ('flash_chord/scripts/evaluate_policy.py', 'tools/pack_submission.py',
                 'tools/preflight.py', 'submit/v2d_submit.py'):
        if sha256_file(KIT / name) != kit_manifest['files'][name]:
            raise ValueError(f'Unedited kit required: {name}')
    source = ROOT / 'upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/evaluation'
    runtime = ROOT / 'data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord/src/flash_chord/evaluation'
    digest = hashlib.sha256()
    inputs = {}
    for name in ('__init__.py', 'harness.py', 'metrics.py', 'rollout.py'):
        p = source / name
        inputs[name] = entry(p)
        # The Linux install normalizes upstream CRLF to LF; no code edits are accepted.
        installed = runtime / name
        if installed.read_bytes() != p.read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError(f'Installed evaluator differs from pinned upstream: {name}')
        inputs['runtime_' + name] = entry(installed)
        digest.update(name.encode() + b'\0' + installed.read_bytes())
    return inputs, digest.hexdigest()


def inspect_export(trajectory: Path, checkpoint_hash: str, episode: int, grid, names: list,
                   eval_code: str) -> tuple[object, np.ndarray]:
    """Admit the evaluator's full schema, provenance, clock and official key grid."""
    table = pq.read_table(trajectory)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    if any('fake' in k.lower() or 'plumbing' in k.lower() for k in meta):
        raise ValueError('FAKE/plumbing trajectories are never admissible')
    expected = {'schema': 'flash_chord_object_trajectories_v1', 'position_unit': 'meter',
                'quaternion_order': 'xyzw', 'episode_index': str(episode),
                'row_order': 'step,environment_id,object_body_id', 'pose_width': '7',
                'checkpoint_sha256': checkpoint_hash,
                'eval_script_sha256': sha256_file(KIT / 'flash_chord/scripts/evaluate_policy.py'),
                'eval_code_sha256': eval_code, 'start_frame': '0', 'world_count': '1',
                'object_body_count': str(len(names)), 'step_count': str(len(grid.frame_index.unique()))}
    for key, value in expected.items():
        if meta.get('flash_chord.' + key) != value:
            raise ValueError(f'Evaluator metadata mismatch: {key}')
    required = ['environment_id', 'step', 'reference_frame', 'time_s', 'object_id',
                'object_body_id', 'object_body_name', 'environment_valid',
                *['achieved_' + c for c in POSE], *['reference_' + c for c in POSE]]
    if any(c not in table.column_names or table[c].null_count for c in required):
        raise ValueError('Missing or null evaluator export columns')
    frame = table.to_pandas().sort_values(['reference_frame', 'environment_id', 'object_body_id'])
    keys = list(zip(frame.environment_id, frame.reference_frame, frame.object_body_id))
    wanted = list(grid[['world_index', 'frame_index', 'object_slot']].itertuples(index=False, name=None))
    if len(set(keys)) != len(keys) or sorted(keys) != sorted(wanted):
        raise ValueError('Duplicate, missing or extra official roster keys')
    if not frame.environment_valid.all():
        raise ValueError('Invalid exported environments')
    if not np.array_equal(frame.object_id, frame.object_body_id):
        raise ValueError('Object grouping differs from official rigid-object slots')
    if any(not isinstance(slot, (int, np.integer)) or slot < 0 or slot >= len(names) or name != names[slot]
           for slot, name in zip(frame.object_body_id, frame.object_body_name)):
        raise ValueError('Official object slot order mismatch')
    fps = number(float(meta['flash_chord.control_fps']), 'control_fps')
    if fps <= 0 or not np.array_equal(frame.step, frame.reference_frame) or not np.allclose(frame.time_s, frame.step / fps, atol=1e-7, rtol=0):
        raise ValueError('Export clock/reset sequence mismatch')
    poses = frame[['achieved_' + c for c in POSE]].to_numpy(float).reshape(-1, len(names), 7)
    require_poses(poses)
    return frame, poses


def require_poses(poses: np.ndarray) -> None:
    if not np.isfinite(poses).all() or not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1., atol=1e-4, rtol=0):
        raise ValueError('Nonfinite poses or nonunit quaternions')


def reference_poses(value: Path, frame, bodies: int) -> np.ndarray:
    """Load the M1 npz pose/clock contract, without reading any labelled dataset."""
    with np.load(value, allow_pickle=False) as data:
        if 'plumbing_only' in data or 'fake' in data:
            raise ValueError('FAKE/plumbing references are never admissible')
        poses = data['pose'].astype(float)
        if poses.ndim == 2:
            poses = poses[:, None, :]
        frames = data['frame_index']
        times = data['time']
    expected_frames = frame.reference_frame.to_numpy().reshape(-1, bodies)[:, 0]
    expected_times = frame.time_s.to_numpy().reshape(-1, bodies)[:, 0]
    if (poses.shape != (len(expected_frames), bodies, 7) or not np.array_equal(frames, expected_frames)
            or times.shape != expected_times.shape or not np.allclose(times - times[0], expected_times, atol=1e-6, rtol=0)):
        raise ValueError('Reference shape/frame/clock differs from export')
    require_poses(poses)
    recorded = frame[['reference_' + c for c in POSE]].to_numpy(float).reshape(poses.shape)
    require_poses(recorded)
    a = poses[..., 3:] / np.linalg.norm(poses[..., 3:], axis=-1, keepdims=True)
    b = recorded[..., 3:] / np.linalg.norm(recorded[..., 3:], axis=-1, keepdims=True)
    b *= np.where(np.sum(a * b, axis=-1, keepdims=True) < 0, -1., 1.)
    if not np.allclose(poses[..., :3], recorded[..., :3], atol=2e-6, rtol=0) or not np.allclose(a, b, atol=2e-6, rtol=0):
        raise ValueError('Independent reference differs from evaluator reference')
    return poses


def admit(adoption: dict, grid, names: list, eval_code: str) -> dict:
    """Verify all scored-adoption clauses and their content-bound producer evidence."""
    episode = adoption['episode']
    inputs = {key: entry(adoption[key]) for key in ('trajectory', 'checkpoint', 'reference', 'admission')}
    admission = read(adoption['admission'])
    if admission.get('schema_version') != 1 or admission.get('episode') != episode or admission.get('plumbing_only'):
        raise ValueError('Admission identity mismatch or FAKE plumbing admission')
    for key in ('trajectory', 'checkpoint', 'reference'):
        if admission['inputs'][key] != inputs[key]:
            raise ValueError(f'Admission does not bind {key} content')
    frame, poses = inspect_export(path(adoption['trajectory']), inputs['checkpoint']['sha256'], episode, grid, names, eval_code)
    reference = reference_poses(path(adoption['reference']), frame, len(names))
    run_entry, official_entry = admission['policy_run'], admission['official_metrics']
    run, official = read(checked(run_entry)), read(checked(official_entry))
    if (run.get('status') != 'completed' or run.get('episode_index') != episode or run.get('split') != 'evaluation'
            or number(run.get('optimizer_updates'), 'optimizer_updates') < 1
            or number(run.get('actor_adam_steps'), 'actor_adam_steps') < 1
            or run.get('checkpoint_sha256') != inputs['checkpoint']['sha256']
            or entry(run['checkpoint']) != inputs['checkpoint']):
        raise ValueError('A learned checkpoint and completed episode policy record are required')
    for key, e in (('trajectory.parquet', inputs['trajectory']), ('official_metrics.json', official_entry)):
        if run.get('artifacts', {}).get(key) != e['sha256']:
            raise ValueError(f'Producer artifact hash mismatch: {key}')
    evaluator = KIT / 'flash_chord/scripts/evaluate_policy.py'
    commands = [c for c in run.get('commands', []) if len(c) > 1 and c[1].endswith('/evaluate_policy.py')]
    if len(commands) != 1 or path(commands[0][1]) != evaluator:
        raise ValueError('Policy producer did not record the kit evaluator command')
    arguments = dict(arg.split('=', 1) for arg in commands[0][2:] if '=' in arg)
    if (path(arguments['evaluation.checkpoint']) != path(adoption['checkpoint'])
            or path(arguments['evaluation.object_trajectories_output']) != path(adoption['trajectory'])
            or path(arguments['evaluation.metrics_output']) != checked(official_entry)
            or arguments.get('evaluation.reset_mode') != 'explicit'):
        raise ValueError('Evaluator command does not bind checkpoint/export/metrics or permits resets')
    require_completion(official, inputs['checkpoint']['sha256'], len(poses))
    m2v, m1 = read(checked(admission['m2v'])), read(checked(admission['m1']))
    require_m2v(m2v)
    require_m1(m1, episode)
    if m1.get('reference_sha256') != inputs['reference']['sha256']:
        raise ValueError('Clause 2 M1 evidence does not bind the adopted reference')
    require_reprojection(admission['achieved_reprojection'], 'Achieved')
    metrics = tracking_metrics(poses, reference)
    for row in metrics:
        require_tracking(row, f'Episode {episode} slot {row["slot"]}')
        require_first(row)
    disclosure = admission['abstract_disclosure']
    checked(disclosure)
    return dict(episode=episode, source='moving policy', **inputs, policy_run=run_entry,
                official_metrics=official_entry, m2v=admission['m2v'], m1=admission['m1'],
                abstract_disclosure=disclosure, tracking=metrics,
                achieved_reprojection=admission['achieved_reprojection'])


def compose_exports(v5: dict, replacements: dict, directory: Path) -> list[dict]:
    """Copy the twenty chosen exports into a fresh composition directory."""
    if not directory.resolve().is_relative_to(SCRATCH) and not directory.resolve().is_relative_to(OUTPUT):
        raise ValueError('Composition output must be under moving-pack outputs or mpack-dry scratch')
    directory.mkdir(parents=True, exist_ok=False)
    exports = []
    for export in v5['exports']:
        episode = int(Path(export['path']).stem.split('_')[1])
        source = path(replacements.get(episode, export['path']))
        destination = directory / f'episode_{episode:06d}.parquet'
        shutil.copyfile(source, destination)
        exports.append(dict(episode=episode, **entry(destination)))
    return exports


def compare_csv(v5: dict, output: Path, adopted: set) -> dict:
    """Require text equality of every unadopted CSV field apart from the commit URL."""
    with path(v5['file']).open(newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; before = list(reader)
    with output.open(newline='') as stream:
        reader = csv.DictReader(stream); after = list(reader)
        if reader.fieldnames != fields:
            raise ValueError('Packed CSV header differs from v5')
    if len(after) != len(before) or [r['row_id'] for r in before] != [r['row_id'] for r in after]:
        raise ValueError('Packed CSV must retain all ordered v5 row IDs')
    unchanged, changed = 0, {}
    for old, new in zip(before, after):
        episode = int(old['row_id'].split('/')[1][1:])
        differences = [k for k in fields if k != 'code_commit_url' and old[k] != new[k]]
        if episode not in adopted:
            if differences:
                raise ValueError(f'Unadopted row changed: {old["row_id"]}: {differences}')
            unchanged += 1
        if differences:
            changed[str(episode)] = changed.get(str(episode), 0) + 1
    return dict(total_rows=len(after), unchanged_rows=unchanged, changed_rows_by_episode=changed,
                comparison='all CSV fields as text, preserving signed zeros; only code_commit_url ignored')


def source_entries() -> list[dict]:
    names = [Path(__file__), Path(__file__).with_name('_moving.py'),
             ROOT / 'src/features/dynamic_feasibility/_metrics.py',
             ROOT / 'src/paths.py', ROOT / 'src/provenance.py',
             ROOT / 'scripts/reproduce/d2_poses_moving.sh', ROOT / 'scripts/reproduce/check_release.sh',
             ROOT / 'docs/chain/moving-m1-protocol.md', ROOT / 'docs/chain/moving-m2-protocol.md']
    return [entry(p) for p in names]


def committed(sources: list[dict], url: str) -> str:
    match = re.fullmatch(r'https://github\.com/[^/]+/[^/]+/commit/([0-9a-f]{40})', url)
    if match is None or 'PLACEHOLDER' in url:
        raise ValueError('A real full-SHA GitHub commit URL is required')
    for source in sources:
        content = subprocess.check_output(['git', 'show', f'{match[1]}:{source["path"]}'], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != source['sha256']:
            raise ValueError(f'Code commit source differs: {source["path"]}')
    return match[1]


def kit_commands(results: Path, output: Path, url: str) -> list[list[str]]:
    python = str(ROOT / 'data/.runtime/kit-env/bin/python')
    return [[python, '-B', str(KIT / 'tools/pack_submission.py'), '--track', 'track_3',
             '--results', str(results), '--code-commit', url, '--eval-script',
             str(KIT / 'flash_chord/scripts/evaluate_policy.py'), '--skip-crosscheck', '--out', str(output)],
            [python, '-B', str(KIT / 'tools/preflight.py'), '--track', 'track_3', '--results', str(results), '--skip-kaggle']]


def pack(request_file: str, code_url: str, output_name: str, dry_run: bool = False) -> Path:
    """Validate first, then compose, pack, preflight and record an immutable manifest."""
    output = path(output_name)
    allowed = SCRATCH if dry_run else OUTPUT
    if not output.is_relative_to(allowed) or output.suffix != '.csv':
        raise ValueError(f'Output must be a CSV under {allowed}; submission/ is forbidden')
    directory = output.parent
    if directory.exists():
        raise FileExistsError('Use a fresh output subdirectory')
    request = read(request_file)
    roster, objects = official_roster()
    adoptions = request['adoptions']
    episodes = [a['episode'] for a in adoptions]
    if len(episodes) != len(set(episodes)) or any(type(ep) is not int or ep not in roster for ep in episodes):
        raise ValueError('Adoptions must be unique scored evaluation episodes; tuning/validation/excluded episodes forbidden')
    if not dry_run and not episodes:
        raise ValueError('No adopted moving episodes; retain D2 v5')
    v5 = frozen_v5()
    evaluation_sources, eval_code = evaluator_inputs()
    admissions = {a['episode']: admit(a, roster[a['episode']], objects[a['episode']].object_names, eval_code) for a in adoptions}
    sources = source_entries()
    if not dry_run:
        committed(sources, code_url)
    elif not re.fullmatch(r'https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}', code_url):
        raise ValueError('Dry pack still requires a syntactically valid full commit URL')
    directory.mkdir(parents=True, exist_ok=False)
    exports = compose_exports(v5, {a['episode']: a['trajectory'] for a in adoptions}, directory / 'results')
    for export in exports:
        ep = export['episode']
        expected = admissions[ep]['trajectory']['sha256'] if ep in admissions else next(
            e['sha256'] for e in v5['exports'] if int(Path(e['path']).stem.split('_')[1]) == ep)
        if export['sha256'] != expected:
            raise ValueError('Export changed between admission and composition')
    commands = kit_commands(directory / 'results', output, code_url)
    for command, name in zip(commands, ('pack.log', 'preflight.log')):
        with (directory / name).open('w') as stream:
            subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    if 'Ready to submit.' not in (directory / 'preflight.log').read_text():
        raise ValueError('Kit preflight not Ready to submit')
    comparison = compare_csv(v5, output, set(episodes))
    provenance = [admissions.get(e['episode'], dict(episode=e['episode'], source='v5 static',
                   trajectory=next(x for x in v5['exports'] if int(Path(x['path']).stem.split('_')[1]) == e['episode']))) for e in exports]
    manifest = dict(kind='pose_insurance.moving_pack', schema_version=1,
                    status='DRY RUN; NOT ADMISSIBLE' if dry_run else 'packed; lead release review required; not uploaded',
                    dry_run=dry_run, code_commit_url=code_url, file=entry(output), request=entry(request_file),
                    v5_manifest=entry(V5_MANIFEST), v5_csv=entry(v5['file']), sources=sources,
                    kit_inputs=[entry(KIT / name) for name in ('tools/pack_submission.py', 'tools/preflight.py',
                                'submit/v2d_submit.py', 'flash_chord/scripts/evaluate_policy.py',
                                'KIT_MANIFEST.json', 'config/tracks.json', 'data/track_3_sample_submission.csv',
                                'data/track_3_evaluation_objects.json')],
                    evaluator_sources=list(evaluation_sources.values()), exports=exports, episodes=provenance,
                    commands=commands, csv_comparison=comparison,
                    preflight_verdict='Ready to submit (Kaggle checks skipped)')
    write(output.with_suffix('.manifest.json'), manifest)
    return output.with_suffix('.manifest.json')


def verify_manifest(name: str) -> None:
    """Re-admit a moving pack for the optional release entry, refusing all dry packs."""
    manifest = read(name)
    if manifest.get('kind') != 'pose_insurance.moving_pack' or manifest.get('dry_run') is not False:
        raise ValueError('Release requires a real admitted moving pack')
    # Verify explicit hash entries only; the immutable v5 parent may itself pin AGENTS.md.
    def visit(value):
        if isinstance(value, dict):
            if 'sha256' in value:
                if Path(value.get('path', '')).name == 'AGENTS.md':
                    raise ValueError('Moving manifest must not pin AGENTS.md')
                checked(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(manifest)
    if manifest['sources'] != source_entries():
        raise ValueError('Moving release source inventory differs from this implementation')
    committed(manifest['sources'], manifest['code_commit_url'])
    v5 = frozen_v5()
    roster, objects = official_roster()
    _, code = evaluator_inputs()
    request = read(checked(manifest['request']))
    adopted = {}
    for adoption in request['adoptions']:
        ep = adoption['episode']
        if type(ep) is not int or ep not in roster or ep in adopted:
            raise ValueError('Invalid or duplicate adopted episode in release')
        adopted[ep] = admit(adoption, roster[ep], objects[ep].object_names, code)
    if not adopted:
        raise ValueError('A moving release requires at least one admitted moving episode')
    expected_episodes = set(roster)
    if len(manifest['exports']) != len(roster) or {e['episode'] for e in manifest['exports']} != expected_episodes:
        raise ValueError('Moving release export roster mismatch')
    for export in manifest['exports']:
        ep = export['episode']
        expected = adopted[ep]['trajectory']['sha256'] if ep in adopted else next(
            e['sha256'] for e in v5['exports'] if int(Path(e['path']).stem.split('_')[1]) == ep)
        if export['sha256'] != expected:
            raise ValueError('Composed export differs from its admitted episode source')
    if manifest['csv_comparison'] != compare_csv(v5, checked(manifest['file']), set(adopted)):
        raise ValueError('Moving release CSV comparison differs')
    expected_provenance = [adopted.get(e['episode'], dict(episode=e['episode'], source='v5 static',
                          trajectory=next(x for x in v5['exports'] if int(Path(x['path']).stem.split('_')[1]) == e['episode'])))
                          for e in manifest['exports']]
    if manifest['episodes'] != expected_provenance:
        raise ValueError('Per-episode manifest provenance differs from re-admission')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='release-repack-', dir=SCRATCH))
    results = work / 'results'
    results.mkdir()
    for export in manifest['exports']:
        shutil.copyfile(checked(export), results / f'episode_{export["episode"]:06d}.parquet')
    for command, label in zip(kit_commands(results, work / 'poses.csv', manifest['code_commit_url']), ('pack', 'preflight')):
        with (work / (label + '.log')).open('w') as stream:
            subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    if sha256_file(work / 'poses.csv') != manifest['file']['sha256']:
        raise ValueError('Moving CSV differs from an official repack of admitted exports')
    print('Moving pack re-admitted; all unadopted v5 rows preserved.')


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--request')
    parser.add_argument('--code-commit')
    parser.add_argument('--out')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-manifest')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    if args.verify_manifest:
        verify_manifest(args.verify_manifest)
        return
    if not all((args.request, args.code_commit, args.out)):
        parser.error('--request, --code-commit and --out are required')
    print(pack(args.request, args.code_commit, args.out, args.dry_run))


if __name__ == '__main__':
    main()
