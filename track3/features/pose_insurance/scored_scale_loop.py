"""Resume scored S1 stereo, serial evaluator attempts and the hybrid v3 package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from track3.provenance import input_file, read_run, sha256_file, write_json
from track3.runs import (ROOT, read_json, producer, artifact, runtime_flag, authorize_episode,
                         heavy_execution, new_run_id, source_hashes)
from .request import Request
from .scene import FEATURE
from ._scored_scale import scaled_scene
from .scored_scale_stereo import prepare

OUTPUT = ROOT / 'outputs/pose_insurance/v3'
EPISODES = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33]
BUDGET = dict(environment_steps=512, worlds=1, replay_minimum=32)
PROTOCOL = ROOT / 'docs/chain/pose-scale-protocol.md'


def append(row: dict) -> None:
    """Durably record each preparation or one-shot evaluator attempt."""
    with (OUTPUT / 'ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def bases(queue: dict) -> dict:
    """Admit each v2 export, its selected policy and its original scene/depth producers."""
    manifest = read_json(queue['base_manifest'])
    selected = read_json('outputs/pose_insurance/v1/evaluation_finished.json')['selected']
    selected = [r for r in selected if r['episode_index'] not in (1, 28)]
    selected += read_json('outputs/pose_insurance/v2/evaluation_finished.json')['selected']
    result = {}
    for row in selected:
        ep = row['episode_index']
        policy = producer(row['run'], 'pose_insurance.run')
        build_entry = next(e for e in policy[1]['inputs'] if '/build-' in e['path'] and e['path'].endswith('/run.json'))
        if input_file(ROOT, build_entry['path']) != build_entry:
            raise ValueError('The v2 scene producer changed')
        build = producer(build_entry['path'], 'pose_insurance.build')
        depth = next(e['path'] for e in build[1]['inputs'] if '/depth/' in e['path'] and e['path'].endswith('/run.json'))
        export = next(e for e in manifest['exports'] if Path(e['path']).name == f'episode_{ep:06d}.parquet')
        if sha256_file(ROOT / export['path']) != export['sha256'] or policy[1]['request']['budget'] != BUDGET:
            raise ValueError('The v2 export or budget differs from its manifest')
        trajectory = artifact(policy, 'trajectory.parquet', {'inputs': []})
        if sha256_file(trajectory) != export['sha256']:
            raise ValueError('The v2 flat export differs from the selected evaluator trajectory')
        result[ep] = dict(base_export=export['path'], base_hash=export['sha256'], base_run=row['run'],
                          base_build=build_entry['path'], depth_run=depth, scene=build[1]['scene'],
                          base_checks=policy[1]['checks'], policy_seconds=policy[1]['policy_seconds'])
    if set(result) != set(EPISODES + [35, 36, 37, 38]):
        raise ValueError('v2 must supply exactly the twenty scored episodes')
    return result


def prepare_plans(queue: dict) -> dict:
    """Compute all fixed stereo estimates before any serial GPU execution."""
    original = bases(queue)
    plans = {}
    for ep in queue['episodes'] + queue['untouched']:
        base = original[ep]
        stereo = prepare(ep, base['depth_run']) if ep in EPISODES else None
        scene = scaled_scene(base['scene'], stereo['s'], stereo['valid']) if stereo else base['scene']
        changed = scene != base['scene']
        reason = ('single object' if stereo is None else 'invalid scale' if not stereo['valid']
                  else 'S1 no-op' if not changed else 'pending physical check')
        plan = dict(base, episode=ep, budget=BUDGET, stereo=stereo, scene=scene,
                    changed=changed, reason=reason, protocol=input_file(ROOT, PROTOCOL),
                    inputs=[input_file(ROOT, base['base_build']), input_file(ROOT, base['base_run'])],
                    sources=source_hashes(FEATURE), stereo_sources=source_hashes(FEATURE.parent / 'stereo_depth'))
        plan['identity'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        path = OUTPUT / f'plans/episode_{ep:06d}.json'
        if path.exists() and read_json(path) != plan:
            raise ValueError('A fixed v3 plan changed; no rerun or recipe adjustment is allowed')
        if not path.exists():
            write_json(path, plan)
            append(dict(stage='plan', episode=ep, identity=plan['identity'], status='completed', changed=changed, reason=reason))
        plans[ep] = plan
        print(json.dumps(dict(stage='plan', episode=ep, s=stereo['s'] if stereo else None,
                              valid=stereo['valid'] if stereo else None, changed=changed, reason=reason)), flush=True)
    estimate = sum(original[ep]['policy_seconds'] for ep, p in plans.items() if p['changed'])
    write_json(OUTPUT / 'plans.json', dict(plans=plans, estimated_gpu_seconds=estimate,
                                        conservative_gpu_seconds=2 * estimate, budget=BUDGET))
    return plans


def completed_attempt(row: dict) -> dict:
    """Resolve a finished attempt without ever retrying a failed or interrupted scene."""
    path = ROOT / row['run']
    if not path.exists():
        return dict(row, status='completed', accepted=False, reason='Interrupted before a run record; no retry')
    record = read_json(path)
    if record['status'] == 'running':
        owner = record.get('owner_pid')
        alive = owner and subprocess.run(['ps', '-p', str(owner), '-o', 'pid='], capture_output=True).returncode == 0
        if alive:
            raise RuntimeError(f"Evaluator attempt is still owned by PID {owner}; wait, do not relaunch")
        return dict(row, status='completed', accepted=False, reason='Interrupted evaluator; no retry')
    if record['status'] != 'completed':
        return dict(row, status='completed', accepted=False, reason=record.get('error', record['status']))
    record = read_run(ROOT, path, kind='pose_insurance.run')
    trajectory = artifact((path, record), 'trajectory.parquet', {'inputs': []})
    checks = record['checks']
    return dict(row, status='completed', accepted=checks['accepted'], checks=checks,
                reason='accepted S1' if checks['accepted'] else checks.get('rejection', 'Physical check failed'),
                trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=sha256_file(trajectory),
                seconds=record['policy_seconds'])


def evaluate(plans: dict, queue: dict) -> dict:
    """Run each changed scene at most once, serially, then freeze accepted/fallback choices."""
    decisions = {}
    rows = [json.loads(line) for line in (OUTPUT / 'ledger.jsonl').read_text().splitlines()]
    attempted, started = 0, time.perf_counter()
    for ep, plan in plans.items():
        if not plan['changed']:
            decisions[ep] = dict(episode=ep, accepted=False, changed=False, reason=plan['reason'], checks=plan['base_checks'])
            continue
        previous = next((r for r in reversed(rows) if r.get('stage') == 'attempt' and r['identity'] == plan['identity']), None)
        if previous:
            result = previous if previous['status'] == 'completed' else completed_attempt(previous)
        else:
            if attempted >= queue['max_gpu_trials'] or time.perf_counter() - started > queue['max_seconds']:
                raise RuntimeError('GPU trial/time budget reached; resume the unchanged queue')
            directory = f'outputs/pose_insurance/run-v3/{new_run_id()}/episode_{ep:06d}'
            row = dict(stage='attempt', identity=plan['identity'], episode=ep, status='started', run=directory + '/run.json')
            append(row)
            command = [str(ROOT / '.runtime/flashchord/venv/bin/python'), '-m',
                       'track3.features.pose_insurance.scored_scale_handler', '--plan',
                       str(OUTPUT / f'plans/episode_{ep:06d}.json'), '--directory', directory]
            with (OUTPUT / f'episode_{ep:06d}.log').open('w') as stream:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
                               stdout=stream, stderr=subprocess.STDOUT)
            result = completed_attempt(row)
            attempted += 1
        if not previous or previous['status'] != 'completed':
            append(result)
        decisions[ep] = dict(result, changed=True)
        print(json.dumps(dict(stage='decision', episode=ep, accepted=result['accepted'], reason=result['reason'],
                              checks=result.get('checks'))), flush=True)
    write_json(OUTPUT / 'decisions.json', decisions)
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', choices=('prepare', 'all', 'pack'), default='all')
    parser.add_argument('--queue', default='track3/features/pose_insurance/queue_v3.json')
    args = parser.parse_args()
    runtime = runtime_flag(FEATURE, 'scored_scale_authorized', 'Scored S1 queue')
    queue = read_json(args.queue)
    if (queue['episodes'] != EPISODES or queue['untouched'] != [35, 36, 37, 38] or queue['budget'] != BUDGET
            or queue['candidate'] != 'S1' or queue['code_commit'] != 'bb8a3f02fac668e86ff7b6a3ad7c3ccaec40050d'):
        raise ValueError('The scored S1 queue must retain the fixed protocol and v2 budget')
    for ep in EPISODES + queue['untouched']:
        authorize_episode(runtime, Request(episode_index=ep), 'scored S1 queue')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'pack':
        plans = {int(k): v for k, v in read_json(OUTPUT / 'plans.json')['plans'].items()}
        decisions = {int(k): v for k, v in read_json(OUTPUT / 'decisions.json').items()}
    else:
        plans = prepare_plans(queue)
        if args.phase == 'prepare':
            return
        decisions = evaluate(plans, queue)
    subprocess.run([str(ROOT / runtime['kit_python']), '-m',
                    'track3.features.pose_insurance.scored_scale_pack'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
