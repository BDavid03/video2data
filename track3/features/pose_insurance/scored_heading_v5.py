"""Build the frozen episode-19 TW scene, run one policy attempt and retain nineteen v4 exports."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from track3.paths import output_path
from track3.provenance import input_file, sha256_file, write_json
from track3.runs import (ROOT, artifact, authorize_episode, checked_entry, new_run_id, producer,
                         read_json, runtime_flag, source_hashes)
from .request import Request
from .scored_stereo_config import BUDGET, EPISODES, UNTOUCHED, FEATURE, KIT, POSE_PROVENANCE
from .scored_stereo_handler import admit_tw, build, environment

QUEUE = 'track3/features/pose_insurance/scored_heading_v5_queue.json'
OUTPUT = ROOT / 'outputs/pose_insurance/heading-v5'
RESULTS = ROOT / 'outputs/pose_insurance/results-v5'
PROTOCOL = 'docs/chain/d2-v5-plan.md'


def append(row: dict) -> None:
    """Append one durable stage record to the heading ledger."""
    with (OUTPUT / 'ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def admission(queue_path: str, cpu: bool = True) -> dict:
    """Admit the dedicated runtime flag, fixed single attempt and exact owner/H3 inputs."""
    runtime = runtime_flag(FEATURE, 'scored_stereo_authorized', 'D2 v5 heading')
    request = Request(episode_index=19, budget=BUDGET)
    authorize_episode(runtime, request, 'D2 v5 heading')
    queue = read_json(queue_path)
    if (queue['candidate'] != 'TW' or queue['episodes'] != [19] or queue['budget'] != BUDGET
            or queue['max_gpu_trials'] != 1 or queue['max_seconds'] != 300):
        raise ValueError('v5 requires one frozen episode-19 TW attempt at the v4 budget')
    for entry in queue['inputs'].values():
        checked_entry(entry, {'inputs': []})
    text = ' '.join((ROOT / queue['inputs']['approval']['path']).read_text().split())
    if '2026-11-03: approve D2 v5 heading for episode 19' not in text or '"all approved"' not in text:
        raise PermissionError('D2 v5 needs the dated owner approval')
    summary = read_json(queue['inputs']['h3_summary']['path'])
    if not summary['gates_passed'] or not all(summary[g]['passed'] for g in ('G1', 'G2', 'G3')):
        raise PermissionError('All H3 gates must pass')
    environment(runtime, cpu=cpu, candidate='TW')
    return queue


def parent_exports(queue: dict) -> tuple:
    """Admit the hashed v4 pack and all twenty unchanged policy exports."""
    manifest = read_json(queue['inputs']['v4_manifest']['path'])
    checked_entry(dict(path=manifest['file'], sha256=manifest['sha256']), {'inputs': []})
    exports = {int(Path(e['path']).stem.split('_')[1]): e for e in manifest['exports']}
    if set(exports) != set((*EPISODES, *UNTOUCHED)) or len(manifest['exports']) != 20:
        raise ValueError('v4 must contain exactly the twenty admitted scored episodes')
    for ep, row in exports.items():
        if sha256_file(ROOT / row['path']) != row['sha256']:
            raise ValueError(f'v4 export changed: {ep}')
    return manifest, exports


def prepare(queue: dict, queue_path: str) -> tuple:
    """Freeze the selected H3 poses and build their equivalent scene on CPU exactly once."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT / 'plan.json'
    if frozen.exists():
        plan = read_json(frozen)
        admit_tw(plan)
    else:
        manifest, exports = parent_exports(queue)
        row = exports[19]
        run_entry = next(e for e in manifest['gpu_runs'] if e['path'] == row['run'])
        policy = producer(checked_entry(run_entry, {'inputs': []}), 'pose_insurance.run', Request(episode_index=19, budget=BUDGET))
        if sha256_file(artifact(policy, 'trajectory.parquet', {'inputs': []})) != row['sha256']:
            raise ValueError('v4 episode 19 export and its policy differ')
        build_entry = next(e for e in policy[1]['inputs'] if '/builds/' in e['path'] and e['path'].endswith('/run.json'))
        parent = producer(checked_entry(build_entry, {'inputs': []}), 'pose_insurance.build')
        if parent[1]['scene'] != policy[1]['scene'] or policy[1]['request']['budget'] != BUDGET:
            raise ValueError('v4 episode 19 policy and task differ')
        report = producer(queue['inputs']['h3_report']['path'], 'pose_insurance.heading_h3_report', allowed=('public_label_diagnostic',))
        if sha256_file(artifact(report, 'summary.json', {'inputs': []})) != queue['inputs']['h3_summary']['sha256']:
            raise ValueError('H3 summary differs from its completed producer')
        summary = read_json(queue['inputs']['h3_summary']['path'])
        pose_run = producer(summary['scored_pose_run'], 'pose_insurance.heading_h3_scored_pose', allowed=POSE_PROVENANCE)
        pose_path = artifact(pose_run, 'episode19.json', {'inputs': []})
        if read_json(pose_path) != summary['episode19']:
            raise ValueError('H3 episode-19 poses differ from the label-free recorded producer')
        poses = summary['episode19']['start_poses']
        scene = deepcopy(parent[1]['scene'])
        for item, pose in zip(scene, poses):
            if item['position'] != pose[:3]:
                raise ValueError('H3 position differs from the admitted v4 task')
            item['quaternion_wxyz'] = [pose[6], *pose[3:6]]
        plan = dict(candidate='TW', episode=19, budget=BUDGET, base_build=build_entry['path'],
                    base_run=row['run'], base_export=row['path'], base_hash=row['sha256'],
                    base_scene=parent[1]['scene'], scene=scene, start_poses=poses,
                    h3_pose_run=str(pose_run[0].relative_to(ROOT)), approval=queue['inputs']['approval'], h3_summary=queue['inputs']['h3_summary'],
                    inputs=[*queue['inputs'].values(), input_file(ROOT, queue_path), run_entry, build_entry, input_file(ROOT, pose_run[0]), input_file(ROOT, pose_path)],
                    protocol=input_file(ROOT, PROTOCOL),
                    sources=source_hashes(FEATURE, extra=(queue_path, FEATURE / 'runtime.json', 'scripts/reproduce/d2_poses_v5.sh')))
        plan['identity'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        admit_tw(plan)
        write_json(frozen, plan)
        append(dict(stage='plan', status='completed', identity=plan['identity']))
    built = OUTPUT / 'build.json'
    if built.exists():
        entry = read_json(built)
        checked_entry(entry, {'inputs': []})
    else:
        path = build(plan, str(frozen.relative_to(ROOT)), Request(episode_index=19, budget=BUDGET))
        entry = input_file(ROOT, path)
        write_json(built, entry)
        append(dict(stage='build', status='completed', identity=plan['identity'], build=entry))
    task = producer(entry['path'], 'pose_insurance.build', allowed=POSE_PROVENANCE)
    if task[1]['plan_identity'] != plan['identity']:
        raise ValueError('Frozen TW build belongs to another plan')
    if not (OUTPUT / 'cpu-inspection.json').exists():
        cpu_inspection(plan, task)
    return plan, entry


def cpu_inspection(plan: dict, task: tuple) -> None:
    """Inspect a task-derived export and demonstrate that the old v4 heading fails the TW scene."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .scored_scale_handler import inspect
    table = pq.read_table(ROOT / plan['base_export'])
    rows = table.to_pylist()
    built = pq.read_table(artifact(task, 'task.parquet', {'inputs': []})).to_pylist()[0]
    for row in rows:
        frame, slot = row['reference_frame'], row['object_body_id']
        for axis, value in zip('xyz', built['object_body_position'][frame][slot]):
            row['achieved_position_'+axis] = value
        wxyz = built['object_body_wxyz'][frame][slot]
        for axis, value in zip('xyzw', [*wxyz[1:], wxyz[0]]):
            row['achieved_quaternion_'+axis] = value
    projection = OUTPUT / 'cpu-task-projection.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), projection)
    parent = producer(plan['base_build'], 'pose_insurance.build')
    previous = pq.read_table(artifact(parent, 'task.parquet', {'inputs': []})).to_pylist()[0]
    allowed = {'object_body_wxyz', 'object_mesh_paths', 'object_urdf_paths'}
    unchanged = [key for key in built if key not in allowed]
    if any(built[key] != previous[key] for key in unchanged):
        raise ValueError('Serialized TW task changed a preserved field')
    contract = inspect(projection, task[1])
    baseline = inspect(ROOT / plan['base_export'], task[1])
    if not contract['accepted'] or baseline['accepted']:
        raise ValueError('TW CPU inspection failed to distinguish the intended and v4 headings')
    write_json(OUTPUT / 'cpu-inspection.json', dict(task_projection=contract, v4_against_tw=baseline,
        projection=input_file(ROOT, projection), equivalence=task[1]['equivalence'],
        frames_preserved=task[1]['task_frames'], unchanged_task_fields=unchanged,
        scope='CPU task/export contract and serialized asset equivalence only; no simulation or scored accuracy evidence'))


def evaluate(plan: dict, build_entry: dict) -> dict:
    """Launch at most one GPU policy attempt and resolve interrupted runs without retrying."""
    from .scored_scale_loop import completed_attempt
    attempt_file = OUTPUT / 'attempt.json'
    if attempt_file.exists():
        attempt = read_json(attempt_file)
        if attempt['identity'] != plan['identity']:
            raise ValueError('An attempt already exists for another plan; no new trial is allowed')
        if attempt.get('owner_pid') != os.getpid():
            alive = subprocess.run(['ps', '-p', str(attempt['owner_pid']), '-o', 'pid='], capture_output=True).returncode == 0
            if alive:
                raise RuntimeError(f"TW attempt is still owned by PID {attempt['owner_pid']}")
    else:
        admit_tw(plan)
        lease_paths = (ROOT / '.runtime/gpu-stage.lock', ROOT.parents[2] / '.runtime/gpu-stage.lock')
        if any(path.exists() for path in lease_paths):
            raise RuntimeError('A GPU lease exists; wait for its owner before launching TW (no attempt consumed)')
        directory = f'outputs/pose_insurance/heading-v5/runs/{new_run_id()}/episode_000019'
        attempt = dict(stage='attempt', status='started', identity=plan['identity'], episode=19,
                       run=directory+'/run.json', owner_pid=os.getpid())
        with attempt_file.open('x') as stream:
            json.dump(attempt, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        append(attempt)
        command = [str(ROOT / '.runtime/flashchord/venv/bin/python'), '-B', '-m',
                   'track3.features.pose_insurance.scored_stereo_handler', '--plan',
                   str(OUTPUT / 'plan.json'), '--directory', directory, '--build', build_entry['path']]
        with (OUTPUT / 'episode_000019.log').open('w') as stream:
            subprocess.run(command, cwd=ROOT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1'),
                           stdout=stream, stderr=subprocess.STDOUT)
    result = completed_attempt(attempt)
    if result['accepted']:
        result['reason'] = 'accepted TW; episode 19 only'
    append(result)
    write_json(OUTPUT / 'decision.json', result)
    return result


def compose(queue: dict, plan: dict, build_entry: dict, decision: dict) -> dict:
    """Copy the new TW trajectory and nineteen byte-identical v4 Parquets into results-v5."""
    if not decision['accepted']:
        raise ValueError('TW failed its single attempt; retain v4, do not pack v5')
    parent, rows = parent_exports(queue)
    RESULTS.mkdir(parents=True, exist_ok=True)
    exports = []
    for ep, original in sorted(rows.items()):
        row = dict(original)
        if ep == 19:
            row.update(source=decision['trajectory'], sha256=decision['trajectory_sha256'], run=decision['run'])
        else:
            row['source'] = original['path']
        destination = RESULTS / f'episode_{ep:06d}.parquet'
        if sha256_file(ROOT / row['source']) != row['sha256']:
            raise ValueError('Selected trajectory changed')
        if destination.exists() and sha256_file(destination) != row['sha256']:
            raise ValueError('Existing v5 export differs from the frozen decision')
        if not destination.exists():
            shutil.copyfile(ROOT / row['source'], destination)
        row['path'] = str(destination.relative_to(ROOT))
        if sha256_file(destination) != row['sha256']:
            raise ValueError('Copied v5 export differs')
        exports.append(row)
    if set(RESULTS.iterdir()) != {ROOT / row['path'] for row in exports}:
        raise ValueError('results-v5 must contain exactly twenty exports')
    task = producer(build_entry['path'], 'pose_insurance.build', allowed=POSE_PROVENANCE)
    policy = producer(decision['run'], 'pose_insurance.run', allowed=POSE_PROVENANCE)
    manifest = dict(version='v5', status='composed; not packed', results_dir=str(RESULTS.relative_to(ROOT)),
        exports=exports, composition=dict(TW=[19], v4=[ep for ep in sorted(rows) if ep != 19]), budget=BUDGET,
        v4_parent=queue['inputs']['v4_manifest'], approval=plan['approval'], h3_summary=plan['h3_summary'],
        tw_plan=input_file(ROOT, OUTPUT / 'plan.json'), build=build_entry, checks=decision['checks'],
        equivalence=task[1]['equivalence'], checkpoint=dict(path=policy[1]['checkpoint'], sha256=policy[1]['checkpoint_sha256']),
        cpu_inspection=input_file(ROOT, OUTPUT / 'cpu-inspection.json'),
        gpu_runs=[input_file(ROOT, decision['run'])],
        inputs=[*plan['inputs'], build_entry, input_file(ROOT, OUTPUT / 'plan.json'), input_file(ROOT, OUTPUT / 'decision.json'),
                *[input_file(ROOT, row['run']) for row in exports]], source_hashes=plan['sources'],
        reproduction_script=input_file(ROOT, 'scripts/reproduce/d2_poses_v5.sh'))
    write_json(OUTPUT / 'composition.json', manifest)
    return manifest


def preflight() -> dict:
    """Record full kit preflight and a separate local result when credentials block it."""
    command = [str(ROOT / '.runtime/kit-env/bin/python'), str(KIT / 'tools/preflight.py'),
               '--track', 'track_3', '--results', str(RESULTS)]
    results = {}
    for label, extra in (('full', []), ('local', ['--skip-kaggle'])):
        log = OUTPUT / f'preflight-{label}.log'
        with log.open('w') as stream:
            result = subprocess.run([*command, *extra], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'), stdout=stream, stderr=subprocess.STDOUT)
        results[label] = dict(command=[*command, *extra], exit_code=result.returncode,
                              ready='Ready to submit.' in log.read_text())
        if label == 'full' and result.returncode == 0:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--queue', default=QUEUE)
    parser.add_argument('--phase', choices=('prepare', 'all'), default='all')
    args = parser.parse_args()
    queue = admission(args.queue)
    plan, build_entry = prepare(queue, args.queue)
    if args.phase == 'prepare':
        print(OUTPUT / 'cpu-inspection.json', flush=True)
        return
    decision = evaluate(plan, build_entry)
    if decision['accepted']:
        compose(queue, plan, build_entry, decision)
        checks = preflight()
    else:
        checks = None
    result = dict(status='completed', accepted=decision['accepted'], decision=decision, preflight=checks,
                  unchanged_exports=19 if decision['accepted'] else 20,
                  next_step='Lead commits and packs v5' if decision['accepted'] else 'Retain v4; no retry')
    write_json(OUTPUT / 'finished.json', result)
    (OUTPUT / 'summary.md').write_text('\n'.join([
        '# D2 v5 frozen TW policy', '',
        f"Episode 19 accepted: {decision['accepted']}. One attempt; no retry.", '',
        f"Physical checks: `{decision.get('checks', {})}`", '',
        'Nineteen v4 exports retained verbatim. CPU equivalence is in `cpu-inspection.json`.', '',
        f"Preflight: `{checks}`", '', result['next_step'], '',
        'Simulation consistency does not establish scored accuracy. The sweeper basin remains ambiguous.', '']))
    print(OUTPUT / 'finished.json', flush=True)


if __name__ == '__main__':
    main()
