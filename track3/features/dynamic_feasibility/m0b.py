"""Frozen M0b queue: resume smoke, throughput probes, selection, training and scoring."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from track3.gpu import GpuLease
from track3.provenance import run_fields, sha256_file, write_json
from track3.runs import recorded
from . import handler, partial_eval
from ._m0b import final_gate, parse_progress, select_recipe, smoke_checks, thresholds, throughput
from .request import ResumeRequest

ROOT = handler.ROOT
OUTPUT = handler.OUTPUT / 'm0b'
QUEUE = handler.FEATURE / 'm0b_queue.json'
STATE = OUTPUT / 'input/state_21575680.safetensors'
POLICY = OUTPUT / 'input/policy_21575680.safetensors'
OPERATIONS = ('smoke', 'probe', 'select', 'train', 'score', 'contact')


def runtime(operation):
    """Refuse an operation before it creates any output unless its M0b flag is true."""
    settings = json.loads((handler.FEATURE / 'runtime.json').read_text())
    if settings.get('M0b', {}).get('execution', {}).get(f'{operation}_authorized') is not True:
        raise PermissionError(f'M0b requires {operation}_authorized')
    return settings


def queue():
    """Admit the frozen protocol and every historical evidence artifact by hash."""
    q = json.loads(QUEUE.read_text())
    if sha256_file(ROOT / q['protocol']) != q['protocol_sha256']:
        raise ValueError('M0b protocol differs from its frozen queue')
    for path, digest in q['evidence'].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f'M0b evidence changed: {path}')
    return q


def inspect_pair(policy, state):
    """Read the fixed scale_by_adam checkpoint layout entirely on CPU, without JAX."""
    import numpy as np
    from safetensors import safe_open
    with safe_open(str(policy), framework='numpy') as saved:
        pm = saved.metadata()
        actor_leaves = len(json.loads(pm['manifest']))
    with safe_open(str(state), framework='numpy') as saved:
        raw = saved.metadata()
        metadata = {k.removeprefix('user.'): v for k, v in raw.items() if k.startswith('user.')}
        manifest = json.loads(raw['manifest'])
        scalar = [i for i, entry in enumerate(manifest) if entry['shape'] == []]
        actor_count, actor_schedule, critic_count, critic_schedule = scalar[:4]
        if actor_count != actor_leaves or metadata['algorithm'] != 'flash_sac' or metadata['checkpoint_kind'] != 'training_state':
            raise ValueError('Unexpected M0b checkpoint layout')
        for key in ('checkpoint_set_id', 'environment_steps'):
            if pm[f'user.{key}'] != metadata[key]:
                raise ValueError('Policy/state are not a complete matched checkpoint pair')
        def tensor(i):
            return saved.get_tensor(manifest[i]['key'])
        with safe_open(str(policy), framework='numpy') as actor:
            policy_manifest = json.loads(pm['manifest'])
            if any(not np.array_equal(tensor(i), actor.get_tensor(entry['key']))
                   for i, entry in enumerate(policy_manifest)):
                raise ValueError('Policy actor differs from its paired training state')
        def parameter_digest(start, end):
            h = hashlib.sha256()
            for i in range(start, end):
                h.update(tensor(i).tobytes())
            return h.hexdigest()
        actor_params = (actor_schedule - actor_count - 1) // 2
        critic_params = (critic_schedule - critic_count - 1) // 2
        summary = {'environment_steps': int(tensor(-2).item()), 'global_update_step': int(tensor(-1).item()),
                   'update_credit': float(tensor(-3).item()), 'curriculum_stage': int(metadata['curriculum_stage']),
                   'voc_scale': float(metadata['voc_scale']), 'actor_adam_steps': int(tensor(actor_count).item()),
                   'critic_adam_steps': int(tensor(critic_count).item()),
                   'actor_parameters_sha256': parameter_digest(0, actor_params),
                   'critic_parameters_sha256': parameter_digest(actor_schedule + 1, actor_schedule + 1 + critic_params),
                   'finite': all(np.isfinite(tensor(i)).all() for i in range(len(manifest))),
                   'checkpoint_set_id': metadata['checkpoint_set_id'],
                   'policy_sha256': sha256_file(policy), 'state_sha256': sha256_file(state)}
        if summary['environment_steps'] != int(metadata['environment_steps']) or summary['global_update_step'] != int(metadata['global_update_step']):
            raise ValueError('Checkpoint tensor and metadata counters disagree')
    return summary


def cpu_inspect(policy, state):
    """Use the existing FlashCHORD safetensors installation with all accelerators hidden."""
    command = [str(handler.PYTHON), '-B', '-m', 'track3.features.dynamic_feasibility.m0b',
               '--inspect', str(policy), str(state)]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT),
                            'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu', 'PYTHONDONTWRITEBYTECODE': '1'},
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def prepare_inputs(q=None):
    """Copy the explicitly approved input pair once and verify both hashes and set identity."""
    q = queue() if q is None else q
    for source, destination, key in ((q['input_state'], STATE, 'state_sha256'), (q['input_policy'], POLICY, 'policy_sha256')):
        if destination.exists():
            if sha256_file(destination) != q[key]:
                raise ValueError('Existing M0b input copy changed')
        else:
            if sha256_file(ROOT / source) != q[key]:
                raise ValueError(f'M0b input digest mismatch: {source}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, destination)
            if sha256_file(destination) != q[key]:
                raise ValueError('M0b copy digest mismatch')
    summary = cpu_inspect(POLICY, STATE)
    path = OUTPUT / 'input/manifest.json'
    record = {'state': handler.entry(STATE), 'policy': handler.entry(POLICY), 'checkpoint': summary}
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError('M0b input manifest changed')
    else:
        write_json(path, record)
    return summary


def identity(q):
    """Bind a new M0b run to its recipe, evidence and current code, independent of flag toggles."""
    paths = [*sorted(handler.FEATURE.glob('*.py')), ROOT / q['protocol'], QUEUE,
             *[ROOT / name for name in q['evidence']], handler.EVALUATOR, handler.KIT_METRIC,
             *sorted((handler.FC / 'src/flash_chord').rglob('*.py')),
             *sorted((handler.FC / 'src/flash_chord/configs').rglob('*.yaml')),
             handler.FC / 'scripts/train_flash_sac.py',
             *[ROOT / 'track3' / name for name in ('gpu.py', 'paths.py', 'runs.py', 'provenance.py')]]
    entries = [handler.entry(path) for path in dict.fromkeys(paths)]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return digest, entries


def gpu_lease(q, operation, allowance=None):
    """Respect the main checkout's lease as well as this checkout's cooperative lock."""
    main_root = next((parent for parent in ROOT.parents if (parent / '.git').is_dir()), ROOT)
    if main_root != ROOT and (main_root / '.runtime/gpu-stage.lock').exists():
        raise RuntimeError('Main-checkout GPU lease already exists')
    return GpuLease(ROOT, q['headroom_mib'], q['admission_allowance_mib'] if allowance is None else allowance,
                    stage=f'dynamic_feasibility_m0b_{operation}')


def request(q, operation, arm=None, selection=None):
    """Build one fixed M0b request without using the D1 sizing or run directories."""
    if operation == 'smoke':
        row = q['smoke']
        label, steps, interval, seconds = 'smoke', row['total_steps'], row['checkpoint_interval'], row['maximum_seconds']
    elif operation == 'probe':
        row = next(row for row in q['probes'] if row['arm'] == arm)
        label, steps, interval, seconds = arm, q['final_steps'], q['checkpoint_interval'], q['probe_seconds']
    else:
        row = selection
        label, steps, interval, seconds = 'train', q['final_steps'], q['checkpoint_interval'], q['train_seconds']
    budget = {'world_count': q['world_count'], 'total_environment_steps': steps,
              'updates_per_collection': row['ratio'], 'capacity': row['capacity'],
              'minimum_size': q['minimum_size'], 'observation_storage_dtype': 'float16',
              'checkpoint_interval': interval, 'save_replay': False,
              'curriculum_thresholds': q['curriculum_thresholds'],
              'learning_rate_schedule_updates': q['lr_horizons'][str(float(row['ratio']))]}
    return ResumeRequest(operation, label, budget, seconds, str(STATE.relative_to(ROOT)))


def training_command(req, directory):
    """Reuse the shared training argv and substitute only the owned M0b progress worker."""
    source = json.loads(QUEUE.read_text())
    task = {'path': ROOT / source['task_parquet'], 'fps': 30.}
    command = handler.training_command(req, task, directory)
    command[3] = 'track3.features.dynamic_feasibility._m0b_worker'
    return command


def ledger(record, key):
    """Append one terminal event per operation identity, including recovered interruptions."""
    path = OUTPUT / 'ledger.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(row['key'] == key and row['identity'] == record['identity'] for row in rows):
        return
    event = {'key': key, 'identity': record['identity'], 'status': record['status'],
             'run': str(record['_path'].relative_to(ROOT))}
    event.update({key: record[key] for key in ('passed', 'admissible', 'complete', 'gpu_seconds',
                                             'ratio', 'capacity', 'updates_done') if key in record})
    with path.open('a') as stream:
        stream.write(json.dumps(event) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def previous(directory, digest):
    """Re-admit an immutable terminal record, or refuse any still-live owner or child."""
    path = directory / 'run.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record['identity'] != digest:
        raise ValueError(f'M0b source or request identity changed: {path}')
    if record['status'] == 'running':
        if any(Path(f'/proc/{record[k]}').exists() for k in ('pid', 'training_pid', 'evaluation_pid') if k in record):
            raise RuntimeError(f'M0b operation still has a live owner: {path}')
        record = {**record, 'status': 'interrupted', 'error': 'Owner exited; no implicit training retry'}
    for entry in record.get('retained', []):
        for name in ('policy', 'state'):
            if sha256_file(ROOT / entry[name]) != entry[f'{name}_sha256']:
                raise ValueError('Retained M0b checkpoint changed')
    return {**record, '_path': path}


def retain_pairs(directory, q):
    """Admit complete pairs and preserve only the boundary and final/latest pair for scoring."""
    training = directory / 'training'
    complete = []
    for state in training.glob('state_*.safetensors'):
        steps = int(state.stem.split('_')[-1])
        policy = state.with_name(f'policy_{steps}.safetensors')
        if policy.exists():
            complete.append((steps, policy, state))
    retained, errors = [], []
    latest = None
    for steps, policy, state in sorted(complete, reverse=True):
        if latest is not None and steps != q['boundary_steps']:
            continue
        try:
            summary = cpu_inspect(policy, state)
            if not summary['finite'] or summary['environment_steps'] != steps:
                raise ValueError('Nonfinite or mislabeled checkpoint pair')
        except Exception as error:
            errors.append({'steps': steps, 'error': str(error)})
            continue
        if latest is None:
            latest = steps
        if steps not in (latest, q['boundary_steps']):
            continue
        target = directory / 'retained'
        target.mkdir(exist_ok=True)
        for source in (policy, state):
            destination = target / source.name
            if not destination.exists():
                shutil.copyfile(source, destination)
            if sha256_file(destination) != sha256_file(source):
                raise ValueError('Retained checkpoint copy differs')
        retained.append({**summary, 'policy': str((target / policy.name).relative_to(ROOT)),
                         'state': str((target / state.name).relative_to(ROOT)),
                         'label': 'boundary, saved at VOC 0.01' if steps == q['boundary_steps'] else 'final/latest'})
        if steps <= q['boundary_steps']:
            break
    return sorted(retained, key=lambda row: row['environment_steps']), errors


def training_trial(req, q, digest, sources):
    """Run one training child under the lease, then enforce CPU admission and record a result."""
    runtime(req.operation)
    directory = OUTPUT / 'trials' / (f'probe-{req.arm}' if req.operation == 'probe' else req.operation)
    old = previous(directory, digest)
    if old:
        ledger(old, directory.name)
        return old
    if req.operation == 'train' and datetime.now(timezone.utc) > datetime.fromisoformat(q['train_launch_deadline_utc']):
        raise RuntimeError('M0b protocol forbids training launch after 05:00 UTC on 4 November')
    if sha256_file(ROOT / q['task_parquet']) != q['task_parquet_sha256']:
        raise ValueError('The frozen M0b shipped task changed')
    before = prepare_inputs(q)
    env = handler.environment()
    record = run_fields(ROOT, 'dynamic_feasibility', f'm0b_{req.operation}', asdict(req),
                        inputs=[QUEUE, ROOT / q['protocol'], handler.FEATURE / 'runtime.json', STATE, POLICY], provenance='public_asset_diagnostic')
    record.update(identity=digest, admitted_sources=sources, pid=os.getpid(), diagnostic_only=True,
                  challenge_episode_access=False, request=asdict(req))
    with recorded(directory, record, hash_artifacts=False):
        command = training_command(req, directory)
        env.update(M0B_OPERATION=req.operation, M0B_CAPACITY=str(req.budget['capacity']),
                   M0B_WORKER_RECORD=str(directory / 'worker.json'))
        worker_path = directory / 'worker.json'
        with gpu_lease(q, req.operation) as lease:
            started = time.monotonic()
            try:
                handler.execute(command, directory / 'training.log', env, time.time() + req.maximum_seconds, record, 'training')
            except RuntimeError as error:
                record['training_error'] = str(error)
            finally:
                record.update(gpu=lease.report(), gpu_seconds=time.monotonic() - started)
            lease.require_headroom()
        record['budget_overrun_seconds'] = max(0., record['gpu_seconds'] - req.maximum_seconds)
        worker = json.loads(worker_path.read_text()) if worker_path.exists() else {'reason': 'crash', 'ring_full_seconds': None}
        rows, compiled = parse_progress((directory / 'training.log').read_text())
        record.update(worker=worker, progress=rows, compile_seconds=compiled,
                      peak_sampled_device_mib=record['gpu']['peak_sampled_total_device_used_mib'],
                      updates_done=rows[-1]['updates'] - before['global_update_step'] if rows else 0)
        record['retained'], record['checkpoint_errors'] = retain_pairs(directory, q)
        if req.operation == 'smoke':
            after = record['retained'][-1] if record['retained'] else None
            checks = smoke_checks(before, after, rows[0]['steps'] if rows else None, record['gpu_seconds']) if after else {'complete_pair': False}
            record.update(checks=checks, passed=all(checks.values()) and worker['reason'] == 'completed' and 'training_error' not in record)
        elif req.operation == 'probe':
            rate = throughput(rows, worker.get('ring_full_seconds'))
            record.update(rate, arm=req.arm, ratio=req.budget['updates_per_collection'], capacity=req.budget['capacity'])
            full = worker.get('ring_full_seconds')
            record['admissible'] = (worker['reason'] in ('ring_full_60_seconds', 'deadline')
                                    and 'training_error' not in record and bool(rate['steady_transitions_per_second'])
                                    and compiled is not None and record['updates_done'] > 0
                                    and record['peak_sampled_device_mib'] <= q['maximum_peak_mib'])
            record['ring_full_dwell_seconds'] = rows[-1]['seconds'] - full if full is not None else None
        else:
            final = record['retained'][-1] if record['retained'] else None
            record['complete'] = bool(final and final['environment_steps'] == q['final_steps'] and final['voc_scale'] == 0.)
            record['completion_label'] = 'complete' if record['complete'] else 'incomplete'
            if record['complete'] and not any(row['environment_steps'] == q['boundary_steps'] for row in record['retained']):
                raise ValueError('Finished training did not retain the exact boundary checkpoint')
    result = {**record, '_path': directory / 'run.json'}
    ledger(result, directory.name)
    return result


def require_smoke(digest):
    """Make every downstream training decision depend on all smoke checks passing."""
    result = previous(OUTPUT / 'trials/smoke', digest)
    if not result or not result.get('passed'):
        raise RuntimeError('M0b stops: smoke is absent, interrupted or failed')
    return result


def probe_summary(rows):
    """Write the compact measured-throughput table without policy-quality selection."""
    lines = ['| Arm | Ratio | Capacity | Admissible | Transitions/s | Compile s | Updates | Peak MiB | Window |',
             '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        lines.append(f"| {row.get('arm', row['request']['arm'])} | {row.get('ratio')} | {row.get('capacity')} | {row.get('admissible', False)} | {row.get('steady_transitions_per_second')} | {row.get('compile_seconds')} | {row.get('updates_done')} | {row.get('peak_sampled_device_mib')} | {row.get('window')} |")
    (OUTPUT / 'probe-summary.md').write_text('\n'.join(lines) + '\n')


def select_operation(q, digest):
    """Persist only the protocol's mechanical selection and every available prediction."""
    runtime('select')
    require_smoke(digest)
    rows = [previous(OUTPUT / f"trials/probe-{row['arm']}", digest) for row in q['probes']]
    if any(row is None for row in rows):
        raise RuntimeError('Run all four probes before selection')
    result = {**select_recipe(rows, maximum_peak=q['maximum_peak_mib']), 'identity': digest,
              'probe_records': [handler.entry(row['_path']) for row in rows]}
    path = OUTPUT / 'selection.json'
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('M0b selection is already frozen with different evidence')
    else:
        with gpu_lease(q, 'select', allowance=0):
            write_json(path, result)
    return result


def load_selection(digest):
    """Re-admit the fixed selection and its probe records before one uninterrupted resume."""
    selected = json.loads((OUTPUT / 'selection.json').read_text())
    if selected['identity'] != digest or selected['status'] != 'selected':
        raise ValueError('M0b has no admissible current selection')
    for entry in selected['probe_records']:
        if handler.entry(ROOT / entry['path']) != entry:
            raise ValueError('A selected throughput record changed')
    return selected


def launch_train(q, digest):
    """Detach the owned training operation and immediately report its PID and log."""
    runtime('train')
    require_smoke(digest)
    load_selection(digest)
    path = OUTPUT / 'launch.json'
    if path.exists():
        print(json.dumps(json.loads(path.read_text()), indent=2))
        return
    if datetime.now(timezone.utc) > datetime.fromisoformat(q['train_launch_deadline_utc']):
        raise RuntimeError('M0b protocol forbids training launch after 05:00 UTC on 4 November')
    command = [str(handler.PYTHON), '-B', '-m', 'track3.features.dynamic_feasibility.m0b', 'train', '--foreground']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log = OUTPUT / 'train-launch.log'
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'},
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    value = {'pid': child.pid, 'argv': command, 'log': str(log.relative_to(ROOT)),
             'finished': 'outputs/dynamic_feasibility/m0b/trials/train/run.json has terminal status; ledger has key train'}
    write_json(path, value)
    print(json.dumps(value, indent=2), flush=True)


def dry_run(operation, q):
    """Print exact argv without checking execution flags, creating outputs or acquiring a GPU."""
    plans = []
    if operation in ('smoke', 'probe', 'train'):
        if operation == 'train':
            if (OUTPUT / 'selection.json').exists():
                selections = [json.loads((OUTPUT / 'selection.json').read_text())]
            else:
                selections = [{'ratio': ratio, 'capacity': capacity} for capacity in (16384, 1048576) for ratio in (.5, 1., 2.)]
        else:
            selections = [None]
        for selected in selections:
            for arm in ([row['arm'] for row in q['probes']] if operation == 'probe' else [None]):
                req = request(q, operation, arm, selected)
                name = f'probe-{arm}' if arm else operation
                directory = OUTPUT / 'trials' / name
                plans.append({'operation': operation, 'arm': req.arm, 'request': asdict(req),
                              'argv': training_command(req, directory), 'deadline_seconds': req.maximum_seconds,
                              'worker_environment': {'M0B_OPERATION': operation, 'M0B_CAPACITY': str(req.budget['capacity']),
                                                     'M0B_WORKER_RECORD': str(directory / 'worker.json')}})
    elif operation == 'score':
        for steps in (q['boundary_steps'], q['final_steps']):
            checkpoint = OUTPUT / f'trials/train/retained/policy_{steps}.safetensors'
            directory = OUTPUT / f'scoring/runs/steps-{steps}/attempt0'
            plans.append({'operation': operation, 'argv': partial_eval.evaluation_command(checkpoint, directory),
                          'note': 'An incomplete run substitutes its last complete pair for the final checkpoint'})
    else:
        plans.append({'operation': operation, 'argv': [str(handler.PYTHON), '-B', '-m', 'track3.features.dynamic_feasibility.m0b', operation],
                      'note': 'Mechanical CPU selection' if operation == 'select' else 'Skipped: force/slip instrumentation exceeds the 20-minute implementation allowance'})
    return plans


def training_result(q, digest):
    """Recover complete pairs after a dead owner without rewriting its historical run record."""
    record = previous(OUTPUT / 'trials/train', digest)
    if not record:
        raise RuntimeError('M0b training has not run')
    if 'retained' not in record:
        path = OUTPUT / 'trials/train/recovery.json'
        if path.exists():
            recovered = json.loads(path.read_text())
        else:
            pairs, errors = retain_pairs(OUTPUT / 'trials/train', q)
            recovered = {'identity': digest, 'retained': pairs, 'checkpoint_errors': errors, 'complete': False,
                         'completion_label': 'incomplete; recovered after interruption'}
            write_json(path, recovered)
        if recovered['identity'] != digest:
            raise ValueError('Recovered M0b source identity changed')
        record.update(recovered)
    if not record['retained']:
        raise RuntimeError('No complete finite M0b pair exists to score')
    return record


def score_context():
    """Load the unchanged prepared task and official metric on the CPU interpreter."""
    import numpy as np
    with np.load(handler.OUTPUT / 'task.npz', allow_pickle=False) as data:
        task = dict(data)
    spec = importlib.util.spec_from_file_location('m0b_kit_auc', handler.KIT_METRIC)
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    return task, kit


def score_preflight():
    """Reuse M0 identities and reproduction, admitting its cached controls by task and artifact hash."""
    from ._metrics import meets_m0
    task, kit = score_context()
    controls_path = handler.OUTPUT / 'm0/jump_hold_controls.json'
    historical = json.loads((handler.OUTPUT / 'm0/run.json').read_text())
    if sha256_file(controls_path) != historical['artifacts']['jump_hold_controls.json']:
        raise ValueError('Cached M0 controls differ from the completed M0 record')
    task_entry = next(entry for entry in historical['inputs'] if entry['path'] == 'outputs/dynamic_feasibility/task.npz')
    if handler.entry(handler.OUTPUT / 'task.npz') != task_entry:
        raise ValueError('Cached controls belong to a different task')
    controls = json.loads(controls_path.read_text())
    result = partial_eval.preflight(task, kit, cached_controls=controls)
    m0 = json.loads((handler.OUTPUT / 'm0/final.json').read_text())
    reproduction = result['checks']['2_reproduction']['details']
    for seed in (0, 1):
        name = f'D1p-seed{seed}'
        directory = handler.OUTPUT / f'm0/runs/{name}/attempt0'
        digest = next(row['sha256'] for row in historical['trials'] if row['name'] == name)
        row = partial_eval.measured_export(directory, task, kit, digest)
        errors = {key: abs(row[key] - m0['rows'][name][key]) for key in ('T', 'Q', 'perpendicular_cm', 'mean_position_cm', 'add_auc')}
        reproduction[name] = {'passed': max(errors.values()) <= 1e-9, 'absolute_errors': errors}
        result['rows'][name] = row
    result['checks']['2_reproduction']['passed'] = all(row['passed'] for row in reproduction.values())
    result['thresholds'] = thresholds(result['rows'])
    result['control_hits'] = {name: [row['j'] for row in controls if meets_m0(row, gate)] for name, gate in result['thresholds'].items()}
    result['checks']['5_controls'] = {'passed': True, 'sha256': sha256_file(controls_path),
                                     'count': len(controls), 'reference_sha256': task_entry['sha256'],
                                     'hits': result['control_hits'], 'recomputed': False}
    return result


def independent_motion(directory, reference):
    """Independently recompute raw T, Q and perpendicular error from the exported positions."""
    import numpy as np
    _, actual = partial_eval.positions(directory)
    a, r = actual - actual[0], reference - reference[0]
    t = np.einsum('ij,ij->', a, r) / np.einsum('ij,ij->', r, r)
    da, dr = np.diff(actual, n=1, axis=0), np.diff(reference, n=1, axis=0)
    da = np.array([da[i:i+15].sum(0) for i in range(375)])
    dr = np.array([dr[i:i+15].sum(0) for i in range(375)])
    q = 1 - np.square(da - dr).sum() / np.square(dr).sum()
    selected = np.linalg.norm(r, axis=1) >= .01
    residual = a[selected] - (np.einsum('ij,ij->i', a[selected], r[selected]) / np.square(r[selected]).sum(1))[:, None] * r[selected]
    return {'T': float(t), 'Q': float(q), 'perpendicular_cm': float(np.linalg.norm(residual, axis=1).mean() * 100)}


def score_final(record):
    """Measure each admitted export, apply all five checks and gate only the final/latest policy."""
    result = json.loads((OUTPUT / 'scoring/preflight.json').read_text())
    task, kit = score_context()
    admissions, identities, audits = {}, {}, {}
    for trial in record['trials']:
        name = trial['name']
        try:
            directory = ROOT / trial['export_directory']
            row = partial_eval.measured_export(directory, task, kit, trial['sha256'])
            row.update(environment_steps=trial['environment_steps'], training_voc_scale=trial['voc_scale'],
                       training_learner_updates=trial['global_update_step'], label=trial['label'])
            result['rows'][name] = row
            admissions[name] = {'passed': True, 'directory': trial['export_directory']}
            independent = independent_motion(directory, task['reference'][:, 0, 0, :3])
            errors = {key: abs(independent[key] - row[key]) for key in independent}
            audits[name] = {'independent': independent, 'absolute_errors': errors, 'passed': max(errors.values()) <= 1e-9}
        except Exception as error:
            result['rows'][name] = {'admission': 'INVALID', 'error': str(error)}
            admissions[name] = {'passed': False, 'error': str(error)}
        identities[name] = {'passed': trial.get('identity_passed', False) and sha256_file(ROOT / trial['checkpoint']) == trial['sha256'],
                            'sha256': trial['sha256'], 'environment_steps': trial['environment_steps'], 'training_voc_scale': trial['voc_scale']}
    result['checks'].update({'3_admission': {'passed': all(r['passed'] for r in admissions.values()), 'details': admissions},
                             '4_checkpoint_identity': {'passed': all(r['passed'] for r in identities.values()), 'details': identities}})
    final_name = max(record['trials'], key=lambda row: row['environment_steps'])['name']
    result.update(final_policy=final_name, complete=record['training']['complete'], recipe=record['recipe'],
                  training_completion=record['training']['completion_label'],
                  training_success_fields=None, training_success_fields_reason='Upstream logging.mode=disabled; no success fields logged',
                  diagnostic_only=True, challenge_episode_access=False, identity=record['identity'])
    result['verdict'] = final_gate(result['rows'][final_name], result['checks'], result['control_hits'], result['thresholds'])
    audit = {'rows': audits, 'all_passed': len(audits) == len(record['trials']) and all(r['passed'] for r in audits.values()),
             'lead_independent_review_required': result['verdict'] in ('GO-1', 'SIGNAL')}
    write_json(OUTPUT / 'audit.json', audit)
    write_json(OUTPUT / 'final.json', result)
    lines = [f"# M0b: {result['verdict']} ({result['training_completion']})", '',
             'One seed; shipped tissue_box_simple only. This is not a challenge result or submission.', '',
             '| Policy | Steps | Training VOC | T | Q | Perpendicular cm | Position cm | Max displacement cm | Orientation deg | ADD-AUC | Updates |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for trial in record['trials']:
        row = result['rows'][trial['name']]
        if row['admission'] == 'INVALID':
            lines.append(f"| {trial['label']} | {trial['environment_steps']} | INVALID |")
            continue
        values = [row.get(key) for key in ('environment_steps', 'training_voc_scale', 'T', 'Q', 'perpendicular_cm',
                   'mean_position_cm', 'max_displacement_cm', 'mean_orientation_deg', 'add_auc', 'training_learner_updates')]
        lines.append(f"| {trial['label']} | " + ' | '.join(str(value) for value in values) + ' |')
    lines += ['', f"Ratio {record['recipe']['ratio']}; capacity {record['recipe']['capacity']}; pinned LR horizon {record['recipe']['lr_horizon']}. "
              'The boundary row is descriptive and was saved at VOC 0.01; the evaluator removes assistance. '
              'Only the final/latest policy is gated. All five M0 validity checks are in final.json; '
              'cached control hashes are checked, and audit.json independently recomputes motion. '
              'Contact force/slip diagnostic skipped: reliable instrumentation exceeds its 20-minute implementation allowance.']
    (OUTPUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    return result


def cpu_measure(phase, record_path):
    """Run M0b measurement with CUDA hidden in the existing measurement interpreter."""
    settings = runtime('score')
    command = [str(ROOT / settings['measurement_python']), '-B', '-m', 'track3.features.dynamic_feasibility.m0b',
               '--measure', phase, '--record', str(record_path)]
    with (record_path.parent / f'{phase}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT), 'CUDA_VISIBLE_DEVICES': '',
                       'PYTHONDONTWRITEBYTECODE': '1'}, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((OUTPUT / ('scoring/preflight.json' if phase == 'preflight' else 'final.json')).read_text())


def score_operation(q, digest, sources):
    """Resume only unexported evaluator attempts and preserve every prior run record."""
    runtime('score')
    training = training_result(q, digest)
    selected = load_selection(digest)
    finished = OUTPUT / 'score-finished.json'
    if finished.exists():
        saved = json.loads(finished.read_text())
        if saved['identity'] != digest or sha256_file(OUTPUT / 'final.json') != saved['final_sha256']:
            raise ValueError('M0b completed score identity changed')
        return json.loads((OUTPUT / 'final.json').read_text())
    env = handler.environment()
    invocations = OUTPUT / 'scoring/invocations'
    for path in invocations.glob('*/run.json'):
        old = json.loads(path.read_text())
        if old['status'] == 'running' and Path(f"/proc/{old['pid']}").exists():
            raise RuntimeError('A score operation is still alive')
    directory = invocations / str(len(list(invocations.glob('*'))))
    record = run_fields(ROOT, 'dynamic_feasibility', 'm0b_score', {'operation': 'score'},
                        inputs=[QUEUE, handler.FEATURE / 'runtime.json', OUTPUT / 'selection.json', training['_path']], provenance='public_asset_diagnostic')
    record.update(identity=digest, admitted_sources=sources, pid=os.getpid(), training={k: training[k] for k in ('complete', 'completion_label')},
                  recipe={'ratio': selected['ratio'], 'capacity': selected['capacity'], 'lr_horizon': q['lr_horizons'][str(float(selected['ratio']))]},
                  trials=[])
    for pair in training['retained']:
        checked = cpu_inspect(ROOT / pair['policy'], ROOT / pair['state'])
        if checked['policy_sha256'] != pair['policy_sha256'] or checked['state_sha256'] != pair['state_sha256']:
            raise ValueError('Training pair changed before evaluation')
        record['trials'].append({**pair, 'name': f"steps-{pair['environment_steps']}", 'source': pair['policy'],
                                 'checkpoint': pair['policy'], 'sha256': pair['policy_sha256'], 'attempts': []})
    with recorded(directory, record, hash_artifacts=False):
        pre = cpu_measure('preflight', directory / 'run.json')
        if not all(row['passed'] for row in pre['checks'].values()):
            raise ValueError('M0b pre-evaluation validity failure')
        with gpu_lease(q, 'score', allowance=4096) as lease:
            started = time.monotonic()
            deadline = time.time() + q['score_seconds']
            for trial in record['trials']:
                partial_eval.evaluate(trial, env, deadline, record, output=OUTPUT / 'scoring', record_path=directory / 'run.json')
                lease.require_headroom()
            record.update(gpu=lease.report(), gpu_seconds=time.monotonic() - started)
        write_json(directory / 'run.json', record)
        result = cpu_measure('final', directory / 'run.json')
    write_json(finished, {'identity': digest, 'final_sha256': sha256_file(OUTPUT / 'final.json'), 'run': str((directory / 'run.json').relative_to(ROOT))})
    ledger({**record, '_path': directory / 'run.json'}, 'score')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', nargs='?', choices=OPERATIONS)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--foreground', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--inspect', nargs=2, metavar=('POLICY', 'STATE'), help=argparse.SUPPRESS)
    parser.add_argument('--measure', choices=('preflight', 'final'), help=argparse.SUPPRESS)
    parser.add_argument('--record', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_pair(*map(Path, args.inspect)), allow_nan=False))
        return
    if args.measure:
        runtime('score')
        q = queue()
        if args.measure == 'preflight':
            write_json(OUTPUT / 'scoring/preflight.json', score_preflight())
        else:
            score_final(json.loads(args.record.read_text()))
        return
    if not args.operation:
        parser.error('Choose an operation')
    if args.dry_run:
        print(json.dumps(dry_run(args.operation, queue()), indent=2))
        return
    runtime(args.operation)
    q = queue()
    digest, sources = identity(q)
    if args.operation == 'smoke':
        result = training_trial(request(q, 'smoke'), q, digest, sources)
        if not result.get('passed'):
            raise RuntimeError('M0b smoke failed; downstream operations are blocked')
    elif args.operation == 'probe':
        require_smoke(digest)
        rows = []
        for arm in q['probes']:
            try:
                row = training_trial(request(q, 'probe', arm['arm']), q, digest, sources)
            except Exception:
                row = previous(OUTPUT / f"trials/probe-{arm['arm']}", digest)
                if row is None:
                    raise
                ledger(row, f"probe-{arm['arm']}")
                row = {**row, **arm, 'admissible': False}
            rows.append({**row, 'arm': arm['arm'], 'ratio': arm['ratio'], 'capacity': arm['capacity']})
            probe_summary(rows)
    elif args.operation == 'select':
        print(json.dumps(select_operation(q, digest), indent=2))
    elif args.operation == 'train':
        if args.foreground:
            require_smoke(digest)
            result = training_trial(request(q, 'train', selection=load_selection(digest)), q, digest, sources)
            print(json.dumps({k: v for k, v in result.items() if k in ('complete', 'completion_label', 'retained')}, indent=2))
        else:
            launch_train(q, digest)
    elif args.operation == 'score':
        print(json.dumps(score_operation(q, digest, sources)['verdict']))
    else:
        path = OUTPUT / 'contact.json'
        if not path.exists():
            write_json(path, {'status': 'skipped', 'reason': 'Force-bearing dwell and relative slip instrumentation exceeds the 20-minute implementation allowance; no GPU rollout started',
                              'identity': digest, 'feeds_gate': False})
        print(path.read_text())


if __name__ == '__main__':
    main()
