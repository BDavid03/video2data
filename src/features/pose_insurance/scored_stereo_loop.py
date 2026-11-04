"""Resume frozen T1 scored scenes, serial policy runs and verified v4 export composition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from src.provenance import input_file, sha256_file, write_json
from src.runs import (ROOT, artifact, authorize_episode, checked_entry, new_run_id,
                         producer, read_json, runtime_flag, source_hashes)
from .scored_stereo_config import FEATURE, KIT, BUDGET, EPISODES, UNTOUCHED
from .request import Request

OUTPUT = ROOT / 'data/outputs/pose_insurance/stereo-placement-scored'
RESULTS = ROOT / 'data/outputs/pose_insurance/results-v4'
PROTOCOL = 'docs/chain/pose-stereo-placement-protocol.md'
QUEUE = 'src/features/pose_insurance/scored_stereo_queue.json'


def base_episode(episode: int, export: dict, record: dict) -> dict:
    """Admit the exact v3 flat export, selected policy and its initial task producer."""
    request = Request(episode_index=episode, budget=BUDGET)
    original = read_json(f'data/outputs/pose_insurance/v3/plans/episode_{episode:06d}.json')
    decision = read_json('data/outputs/pose_insurance/v3/decisions.json')[str(episode)]
    policy = producer(decision['run'] if decision['accepted'] else original['base_run'], 'pose_insurance.run', request)
    record['inputs'].append(input_file(ROOT, policy[0]))
    trajectory = artifact(policy, 'trajectory.parquet', record)
    if sha256_file(trajectory) != export['sha256'] or sha256_file(ROOT / export['path']) != export['sha256']:
        raise ValueError('v3 flat export does not match its selected policy')
    build_entry = next(entry for entry in policy[1]['inputs']
                       if '/build-' in entry['path'] and entry['path'].endswith('/run.json'))
    checked_entry(build_entry, record)
    build = producer(build_entry['path'], 'pose_insurance.build', request)
    if policy[1]['scene'] != build[1]['scene'] or policy[1]['request']['budget'] != BUDGET:
        raise ValueError('v3 task scene or budget differs from its policy record')
    return dict(episode=episode, budget=BUDGET, base_export=export['path'], base_hash=export['sha256'],
                base_run=str(policy[0].relative_to(ROOT)), base_build=build_entry['path'],
                base_scene=build[1]['scene'], base_checks=policy[1]['checks'], original_plan=original)


def measure_offset(base: dict, record: dict) -> tuple:
    """Read frame-zero stereo and v3 cluster unions using only evaluation video producers."""
    import numpy as np
    from .length_diagnostic import frame_arrays
    from ._stereo_placement import stereo_offset
    episode = base['episode']
    ids = [row['clusters'] for row in base['base_scene']]
    details = dict(cluster_ids=ids, stereo_valid_pixels=[0, 0])
    if any(row.get('merged_cluster_ids') is not None for row in base['base_scene']) or any(i is None for i in ids):
        return None, 'slot without its own observed cluster (merged/default)', details
    if set(ids[0]) & set(ids[1]):
        return None, 'merged cluster shared by both slots', details
    request = Request(episode_index=episode, budget=BUDGET)
    original = base['original_plan']
    old_build = producer(original['base_build'], 'pose_insurance.build', request)
    record['inputs'].append(input_file(ROOT, old_build[0]))
    entry = next(e for e in old_build[1]['inputs'] if '/mesh_extents/' in e['path']
                 and '/build/' not in e['path'] and e['path'].endswith('/run.json'))
    checked_entry(entry, record)
    measure = producer(entry['path'], 'mesh_extents.measure', request)
    depth = producer(original['depth_run'], 'depth.generate', request)
    record['inputs'].append(input_file(ROOT, depth[0]))
    if measure[1]['request']['depth_run'] != original['depth_run'] or measure[1]['frames'][0]['frame_index'] != 0:
        raise ValueError('Observed masks do not use the exact v3 frame-zero depth producer')
    with np.load(artifact(measure, 'labels_000000.npz', record)) as arrays:
        labels = arrays['label'].copy()
    scale_path = f'data/outputs/pose_insurance/v3/stereo_{episode:06d}.json'
    record['inputs'].append(input_file(ROOT, scale_path))
    scale = read_json(scale_path)
    for entry in scale['anchors']:
        checked_entry(entry, record)
        anchor = producer(entry['path'], 'stereo_depth.anchor', request)
        if any(f['frame_index'] == 0 for f in anchor[1]['frames']):
            break
    else:
        return None, 'no frame-zero stereo run', details
    if anchor[1]['request']['depth_run'] != original['depth_run']:
        raise ValueError('Stereo anchor uses another depth producer')
    stereo = producer(anchor[1]['request']['stereo_run'], 'stereo_depth.generate', request)
    record['inputs'].append(input_file(ROOT, stereo[0]))
    if stereo[1]['grid'] != depth[1]['preprocessing_artifact_sha256']:
        raise ValueError('Stereo and MoGe grid hashes differ')
    arrays = frame_arrays(stereo, record)
    with np.load(artifact(depth, 'preprocessing.npz', record, depth[1]['preprocessing_artifact_sha256'])) as grid:
        same_k = np.allclose(arrays['rgb_intrinsics_pixels'], grid['input_intrinsics_pixels'], atol=1e-5)
    if (int(arrays['frame_index']) != 0 or int(arrays['pts']) != 0
            or arrays['rgb_depth_m'].shape != labels.shape or not same_k):
        raise ValueError('Stereo timestamps, shape or intrinsics differ from the frame-zero masks')
    masks = [np.isin(labels, np.asarray(i) + 1) for i in ids]
    displacement, counts = stereo_offset(arrays, masks, np.array(measure[1]['frames'][0]['plane']['normal']))
    details.update(stereo_valid_pixels=counts, grid=stereo[1]['grid'],
                   measure_run=str(measure[0].relative_to(ROOT)),
                   stereo_run=str(stereo[0].relative_to(ROOT)))
    return displacement, 'fewer than 300 valid stereo pixels on an object' if displacement is None else None, details


def resume_plans(ledger) -> dict | None:
    """Re-admit immutable plans without rebuilding them when unrelated sources change."""
    from .scale_loop import append
    saved = OUTPUT / 'plans.json'
    if saved.exists():
        plans = {int(ep): plan for ep, plan in read_json(saved)['plans'].items()}
        if set(plans) != set((*EPISODES, *UNTOUCHED)):
            raise ValueError('Frozen plans do not cover exactly the scored roster')
        current_sources = source_hashes(FEATURE)
        differences = {}
        for ep, plan in plans.items():
            payload = {key: value for key, value in plan.items() if key != 'identity'}
            if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != plan['identity']:
                raise ValueError('Frozen plan identity differs from its contents')
            if read_json(OUTPUT / f'plans/episode_{ep:06d}.json') != plan:
                raise ValueError('Per-episode plan differs from the frozen inventory')
            for entry in plan['inputs']:
                checked_entry(entry, {'inputs': []})
            differences[ep] = {path: dict(frozen=digest, current=current_sources.get(path))
                               for path, digest in plan['sources'].items() if current_sources.get(path) != digest}
        write_json(OUTPUT / 'resume_source_differences.json', differences)
        append(ledger, dict(stage='all_plans_frozen', status='resumed',
                            identities={ep:p['identity'] for ep,p in plans.items()}))
        return plans
    return None


def prepare(queue: dict, ledger) -> dict:
    """Freeze every clearance and fallback before any policy attempt."""
    import numpy as np
    from ._stereo_placement import scene_placement
    from .scale_loop import append
    saved = resume_plans(ledger)
    if saved is not None:
        return saved
    train = read_json('data/outputs/pose_insurance/stereo-placement/summary.json')
    if train['verdict'] != 'T1 goes to the scored stage' or not train['gates']['passed'] or not all(train['integrity'].values()):
        raise PermissionError('T1 scored stage requires all train gates and integrity controls')
    manifest = read_json(queue['base_manifest'])
    exports = {int(Path(e['path']).stem.split('_')[1]): e for e in manifest['exports']}
    if set(exports) != set((*EPISODES, *UNTOUCHED)):
        raise ValueError('v3 manifest must contain exactly the twenty scored episodes')
    plans = {}
    for episode in sorted(exports):
        record = {'inputs': [input_file(ROOT, p) for p in (PROTOCOL, QUEUE, queue['base_manifest'],
                    'data/outputs/pose_insurance/stereo-placement/summary.json',
                    f'data/outputs/pose_insurance/v3/plans/episode_{episode:06d}.json',
                    'data/outputs/pose_insurance/v3/decisions.json')]}
        base = base_episode(episode, exports[episode], record)
        if episode in EPISODES:
            displacement, reason, observations = measure_offset(base, record)
            scene, clearance = scene_placement(base['base_scene'], displacement, reason)
        else:
            displacement, observations, scene = None, {}, base['base_scene']
            clearance = dict(status='unchanged', reason='outside the thirteen-episode T1 roster', factor=None)
        base.pop('original_plan')
        s1_length = float(np.linalg.norm(np.array(base['base_scene'][1]['position'][:2]) - base['base_scene'][0]['position'][:2])) if len(scene) > 1 else None
        plan = dict(base, scene=scene, changed=scene != base['base_scene'], clearance=clearance,
                    raw_offset_m=None if displacement is None else displacement.tolist(),
                    stereo_length_m=None if displacement is None else float(np.linalg.norm(displacement)),
                    s1_length_m=s1_length, observations=observations, inputs=record['inputs'],
                    sources=source_hashes(FEATURE), protocol=input_file(ROOT, PROTOCOL))
        plan['identity'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        path = OUTPUT / f'plans/episode_{episode:06d}.json'
        if path.exists() and read_json(path) != plan:
            raise ValueError('A frozen T1 plan changed; review instead of rerunning it')
        if not path.exists():
            write_json(path, plan)
            append(ledger, dict(stage='plan', episode=episode, identity=plan['identity'], status='completed',
                                changed=plan['changed'], clearance=clearance))
        plans[episode] = plan
    write_json(OUTPUT / 'plans.json', dict(plans=plans, budget=BUDGET))
    append(ledger, dict(stage='all_plans_frozen', status='completed', identities={ep:p['identity'] for ep,p in plans.items()}))
    return plans


def evaluate(plans: dict, queue: dict, rows: list, ledger) -> tuple:
    """Reuse one-shot attempt resolution while running changed scenes serially."""
    from .scored_scale_handler import inspect
    from .scored_scale_loop import completed_attempt
    from .scale_loop import append
    decisions, executed, resumed, started = {}, 0, 0, time.perf_counter()
    for episode, plan in plans.items():
        if not plan['changed']:
            policy = producer(plan['base_run'], 'pose_insurance.run')
            task = producer(plan['base_build'], 'pose_insurance.build')
            checks = inspect(artifact(policy, 'trajectory.parquet', {'inputs': []}), task[1])
            decisions[episode] = dict(episode=episode, accepted=False, reason=plan['clearance']['reason'], checks=checks)
            continue
        previous = next((row for row in reversed(rows) if row.get('stage') == 'attempt' and row['identity'] == plan['identity']), None)
        if previous:
            owner = previous.get('owner_pid')
            if previous['status'] != 'completed' and owner and owner != os.getpid():
                alive = subprocess.run(['ps', '-p', str(owner), '-o', 'pid='], capture_output=True).returncode == 0
                if alive:
                    raise RuntimeError(f'T1 loop attempt is still owned by PID {owner}')
            result = completed_attempt(previous)
            resumed += 1
        else:
            execution_files = ('scored_stereo_handler.py', 'scored_scale_handler.py', 'handler.py',
                               '_scored_scale.py', '_scene.py', 'scene.py', 'request.py')
            paths = [f'src/features/pose_insurance/{name}' for name in execution_files]
            paths += [f'src/{name}' for name in ('gpu.py', 'runs.py', 'paths.py', 'provenance.py')]
            if any(sha256_file(ROOT / path) != plan['sources'][path] for path in paths):
                raise ValueError('Policy execution sources changed after the plans were frozen')
            if executed >= queue['max_gpu_trials'] or time.perf_counter() - started > queue['max_seconds']:
                raise RuntimeError('GPU trial/time budget reached; resume the unchanged queue')
            directory = f'data/outputs/pose_insurance/stereo-placement-scored/runs/{new_run_id()}/episode_{episode:06d}'
            attempt = dict(stage='attempt', status='started', identity=plan['identity'], episode=episode,
                           run=directory + '/run.json', owner_pid=os.getpid())
            append(ledger, attempt)
            command = [str(ROOT / 'data/.runtime/flashchord/venv/bin/python'), '-B', '-m',
                       'src.features.pose_insurance.scored_stereo_handler', '--plan',
                       str(OUTPUT / f'plans/episode_{episode:06d}.json'), '--directory', directory]
            with (OUTPUT / f'episode_{episode:06d}.log').open('w') as stream:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1'),
                               stdout=stream, stderr=subprocess.STDOUT)
            result = completed_attempt(attempt)
            executed += 1
        if result['accepted']:
            result['reason'] = 'accepted T1'
        if previous is None or previous['status'] != 'completed':
            append(ledger, result)
        decisions[episode] = result
        print(json.dumps(dict(episode=episode, accepted=result['accepted'], reason=result['reason'],
                              checks=result.get('checks'))), flush=True)
    write_json(OUTPUT / 'decisions.json', decisions)
    return decisions, executed, resumed


def compose(plans: dict, decisions: dict, queue: dict) -> dict:
    """Copy accepted T1 or exact v3 exports into a complete verified twenty-file directory."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    exports, composition = [], dict(T1=[], v3=[])
    for episode, plan in sorted(plans.items()):
        decision = decisions[episode]
        source = ROOT / (decision['trajectory'] if decision['accepted'] else plan['base_export'])
        expected = decision['trajectory_sha256'] if decision['accepted'] else plan['base_hash']
        destination = RESULTS / f'episode_{episode:06d}.parquet'
        if sha256_file(source) != expected:
            raise ValueError('Selected export source hash changed')
        if destination.exists() and sha256_file(destination) != expected:
            raise ValueError('An existing v4 export differs from the frozen decision')
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha256_file(destination) != expected:
            raise ValueError('Composed export differs from its selected source')
        exports.append(dict(path=str(destination.relative_to(ROOT)), source=str(source.relative_to(ROOT)), sha256=expected,
                            run=decision['run'] if decision['accepted'] else plan['base_run']))
        composition['T1' if decision['accepted'] else 'v3'].append(episode)
    if set(RESULTS.iterdir()) != {ROOT / row['path'] for row in exports} or len(exports) != 20:
        raise ValueError('results-v4 must contain exactly twenty episode Parquets')
    manifest = dict(version='v4', status='composed; not packed', results_dir=str(RESULTS.relative_to(ROOT)),
                    exports=exports, composition=composition, budget=BUDGET,
                    gpu_runs=[input_file(ROOT, d['run']) for d in decisions.values()
                              if 'run' in d and (ROOT / d['run']).is_file()],
                    inputs=[input_file(ROOT, path) for path in (PROTOCOL, QUEUE, queue['base_manifest'], OUTPUT / 'plans.json', OUTPUT / 'decisions.json',
                              *dict.fromkeys(row['run'] for row in exports))],
                    source_hashes=source_hashes(FEATURE),
                    reproduction_script=input_file(ROOT, 'scripts/reproduce/d2_poses_v4.sh'))
    write_json(OUTPUT / 'composition.json', manifest)
    return manifest


def summarize(plans: dict, decisions: dict, manifest: dict, executed: int, resumed: int, preflight: dict) -> None:
    """Publish the rendered result and the complete machine-readable evidence."""
    from .scored_stereo_report import markdown
    (OUTPUT / 'summary.md').write_text(markdown(plans, decisions, manifest, executed, resumed, preflight, EPISODES))
    write_json(OUTPUT / 'summary.json', dict(executed=executed, resumed=resumed, decisions=decisions,
               composition=manifest['composition'], plans=plans, preflight=preflight))
    write_json(OUTPUT / 'finished.json', dict(status='completed', executed=executed, resumed=resumed,
               composition=manifest['composition'], preflight=preflight))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', choices=('prepare', 'all'), default='all')
    args = parser.parse_args()
    runtime = runtime_flag(FEATURE, 'scored_stereo_authorized', 'T1 scored stage')
    queue = read_json(QUEUE)
    if (queue['episodes'] != list(EPISODES) or queue['untouched'] != list(UNTOUCHED)
            or queue['budget'] != BUDGET or queue['candidate'] != 'T1'
            or not 1 <= queue['max_gpu_trials'] <= 12 or not 1 <= queue['max_seconds'] <= 7200):
        raise ValueError('Scored T1 queue differs from the fixed roster or budget')
    for ep in (*EPISODES, *UNTOUCHED):
        authorize_episode(runtime, Request(episode_index=ep, budget=BUDGET), 'T1 scored stage')
    from .scored_stereo_handler import environment
    environment(runtime, cpu=True)
    from .shrink_loop import resume_ledger
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, ledger = resume_ledger(OUTPUT)
    plans = prepare(queue, ledger)
    if args.phase == 'prepare':
        print(OUTPUT / 'plans.json')
        return
    decisions, executed, resumed = evaluate(plans, queue, rows, ledger)
    manifest = compose(plans, decisions, queue)
    command = [str(ROOT / 'data/.runtime/kit-env/bin/python'), str(KIT / 'tools/preflight.py'),
               '--track', 'track_3', '--results', str(RESULTS)]
    with (OUTPUT / 'preflight.log').open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
                                stdout=stream, stderr=subprocess.STDOUT)
    preflight = dict(command=command, exit_code=result.returncode,
                     ready='Ready to submit.' in (OUTPUT / 'preflight.log').read_text())
    if result.returncode:
        local_command = [*command, '--skip-kaggle']
        with (OUTPUT / 'preflight-local.log').open('w') as stream:
            local = subprocess.run(local_command, cwd=ROOT,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
                                   stdout=stream, stderr=subprocess.STDOUT)
        preflight['local'] = dict(exit_code=local.returncode,
            ready='Ready to submit.' in (OUTPUT / 'preflight-local.log').read_text())
    summarize(plans, decisions, manifest, executed, resumed, preflight)
    print(OUTPUT / 'summary.md')


if __name__ == '__main__':
    main()
