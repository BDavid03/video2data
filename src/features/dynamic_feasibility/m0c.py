"""Fresh seed-1 M0b recipe, fixed checkpoint evaluation, and isolated run records."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import time

from src.gpu import GpuLease
from src.provenance import run_fields, sha256_file, write_json
from src.runs import CACHE_FOLDERS, recorded
from . import handler, m0b, partial_eval
from ._m0b import parse_progress
from .request import Request

ROOT = handler.ROOT
OUTPUT = handler.OUTPUT / 'm0c'
QUEUE = handler.FEATURE / 'm0c_queue.json'
TRAIN = OUTPUT / 'trials/train'
MODULE = 'src.features.dynamic_feasibility.m0c'


def queue():
    """Admit the committed protocol, fixed recipe and shipped task identity."""
    q = json.loads(QUEUE.read_text())
    old = m0b.queue()
    if sha256_file(ROOT / q['protocol']) != q['protocol_sha256']:
        raise ValueError('M0c protocol changed')
    if q['seed'] != 1 or q['train_seconds'] != 4500:
        raise ValueError('M0c requires seed 1 and the fixed 75-minute deadline')
    for key in ('world_count', 'minimum_size', 'n_step', 'warmup_action', 'checkpoint_interval',
                'curriculum_thresholds', 'final_steps', 'task_parquet', 'task_parquet_sha256'):
        if q[key] != old[key]:
            raise ValueError(f'M0c differs from the frozen M0b recipe: {key}')
    selected = json.loads((m0b.OUTPUT / 'selection.json').read_text())
    if (q['ratio'], q['capacity'], q['lr_horizon']) != (selected['ratio'], selected['capacity'], old['lr_horizons']['2.0']):
        raise ValueError('M0c must use M0b\'s selected recipe')
    if q['retain_steps'] != [16154624, 32814080, 66763776]:
        raise ValueError('M0c checkpoint schedule changed')
    if (any(step % q['world_count'] for step in q['retain_steps'])
            or any(step % q['checkpoint_interval'] for step in q['retain_steps'][:-1])):
        raise ValueError('M0c checkpoints must lie on both save and collection grids')
    if sha256_file(ROOT / q['task_parquet']) != q['task_parquet_sha256']:
        raise ValueError('The shipped tissue-box task changed')
    return q


def runtime(operation, measurement=False):
    """Check M0c authorization without changing any historical operation flag."""
    settings = json.loads((handler.FEATURE / 'runtime.json').read_text())
    if settings.get('M0c', {}).get('execution', {}).get(f'{operation}_authorized') is not True:
        raise PermissionError(f'M0c requires {operation}_authorized')
    python = ROOT / settings['measurement_python'] if measurement else handler.PYTHON
    if Path(sys.executable).resolve() != python.resolve():
        raise RuntimeError('Use the configured interpreter for this M0c operation')
    return settings


def environment(create=True):
    """Route caches, configuration and temporary writes solely into this operation tree."""
    folders = {key: OUTPUT / 'runtime' / folder.lstrip('.') for key, folder in CACHE_FOLDERS.items()}
    folders.update({key: OUTPUT / 'runtime' / name for key, name in {
        'WARP_CACHE_PATH': 'warp', 'JAX_COMPILATION_CACHE_DIR': 'jax',
        'FLASH_CHORD_CACHE_DIR': 'flash-chord', 'NUMBA_CACHE_DIR': 'numba',
        'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'data', 'XDG_STATE_HOME': 'state',
        'WANDB_DIR': 'wandb', 'WANDB_CACHE_DIR': 'wandb-cache',
        'WANDB_CONFIG_DIR': 'wandb-config', 'WANDB_DATA_DIR': 'wandb-data',
    }.items()})
    env = {**os.environ, **{key: str(path) for key, path in folders.items()},
           'PYTHONPATH': os.pathsep.join((str(ROOT), str(handler.FC / 'src'))),
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
           'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'JAX_ENABLE_COMPILATION_CACHE': 'true',
           'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS': '0', 'WANDB_MODE': 'disabled',
           'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HYDRA_FULL_ERROR': '1'}
    if create:
        if OUTPUT.resolve() != OUTPUT or any(not path.resolve().is_relative_to(OUTPUT) for path in folders.values()):
            raise ValueError('M0c output/cache paths must stay inside this worktree')
        for path in set(folders.values()):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        tempfile.tempdir = env['TMPDIR']
    return env


def check_m0b_identity():
    """Report raw M0b admission separately from unchanged historical manifest admission."""
    current, entries = m0b.identity(m0b.queue())
    historical = json.loads((m0b.OUTPUT / 'trials/train/run.json').read_text())
    old_entries = historical['admitted_sources']
    for entry in old_entries:
        if handler.entry(ROOT / entry['path']) != entry:
            raise ValueError(f'Historical M0b source changed: {entry["path"]}')
    digest = hashlib.sha256(json.dumps(old_entries, sort_keys=True).encode()).hexdigest()
    if digest != historical['identity']:
        raise ValueError('Historical M0b manifest does not reproduce its identity')
    # The immutable M0b glob includes this newly requested module.
    added = [entry['path'] for entry in entries if entry['path'] not in {e['path'] for e in old_entries}]
    if added not in ([], [str(Path(__file__).relative_to(ROOT))]):
        raise ValueError(f'Unexpected additions to M0b identity: {added}')
    m0b.previous(m0b.OUTPUT / 'trials/train', digest)
    m0b.load_selection(digest)
    try:
        m0b.previous(m0b.OUTPUT / 'trials/train', current)
        raw_admits = True
    except ValueError:
        raw_admits = False
    return {'historical_manifest_admits': True, 'raw_identity_admits': raw_admits,
            'historical_identity': digest, 'raw_identity': current, 'added_sources': added,
            'runtime_hashed_by_identity': False, 'read_only': True}


def identity(q):
    """Bind M0c to current sources, protocol, queue and its fixed historical comparison."""
    _, sources = m0b.identity(m0b.queue())
    sources += [handler.entry(path) for path in (QUEUE, ROOT / q['protocol'],
                ROOT / q['task_parquet'], m0b.OUTPUT / 'selection.json', m0b.OUTPUT / 'final.json')]
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(), sources


def request(q):
    """Use the shared fresh-training request, rather than seed-0-only ResumeRequest."""
    budget = {'world_count': q['world_count'], 'total_environment_steps': q['final_steps'],
              'updates_per_collection': q['ratio'], 'capacity': q['capacity'],
              'minimum_size': q['minimum_size'], 'observation_storage_dtype': 'float16',
              'checkpoint_interval': q['checkpoint_interval'], 'save_replay': False,
              'curriculum_thresholds': q['curriculum_thresholds']}
    return Request('D1', q['seed'], budget, q['train_seconds'])


def training_command(q):
    """Reuse shared argv construction and explicitly pin every fresh-run difference."""
    command = handler.training_command(request(q), {'path': ROOT / q['task_parquet'], 'fps': 30.}, TRAIN)
    command[3:4] = [MODULE, '--worker']
    return command + ['training.resume=false', 'training.checkpoint_path=null',
                      f'training.optimization.learning_rate_schedule_updates={q["lr_horizon"]}',
                      'training.exploration.warmup_action="policy"', 'training.replay.n_step=3']


def dry_run(operation, q):
    """Print exact launch/evaluator argv without creating files or taking a GPU lease."""
    env = environment(create=False)
    steps = q['retain_steps']
    if operation == 'score' and (TRAIN / 'run.json').exists():
        training = m0b.previous(TRAIN, identity(q)[0])
        if 'retained' in training:
            steps = [pair['environment_steps'] for pair in training['retained']]
    plans = [training_command(q)] if operation == 'train' else [
        partial_eval.evaluation_command(TRAIN / f'retained/policy_{step}.safetensors',
            OUTPUT / f'scoring/runs/steps-{step}/attempt0') for step in steps]
    return {'operation': operation, 'argv': plans, 'deadline_seconds': q[f'{operation}_seconds'],
            'environment': {key: value for key, value in env.items()
                            if key not in os.environ or os.environ[key] != value},
            'retention': 'No D1 temporary_checkpoints pruning is invoked; all generated pairs remain.',
            'incomplete': 'Score reached fixed rows plus the last complete finite pair, without duplicates.'}


def retain_pairs(q):
    """Keep reached fixed checkpoints and the last complete finite pair, with matching hashes."""
    candidates = {int(path.stem.rsplit('_', 1)[1]): path for path in (TRAIN / 'training').glob('state_*.safetensors')
                  if path.with_name(path.name.replace('state_', 'policy_', 1)).exists()}
    retained, errors = {}, []
    wanted = set(q['retain_steps'])
    for step in sorted(candidates, reverse=True):
        if retained and step not in wanted:
            continue
        state = candidates[step]
        policy = state.with_name(f'policy_{step}.safetensors')
        try:
            summary = m0b.cpu_inspect(policy, state)
            if not summary['finite'] or summary['environment_steps'] != step:
                raise ValueError('Nonfinite or mislabeled checkpoint pair')
            expected = {16154624: (3, .25), 32814080: (7, .01), 66763776: (10, 0.)}
            if step in expected and (summary['curriculum_stage'], summary['voc_scale']) != expected[step]:
                raise ValueError('Prescribed checkpoint has unexpected saved curriculum/VOC')
        except Exception as error:
            errors.append({'steps': step, 'error': str(error)})
            continue
        target = TRAIN / 'retained'
        target.mkdir(exist_ok=True)
        for source in (policy, state):
            destination = target / source.name
            if not destination.exists():
                shutil.copyfile(source, destination)
            if sha256_file(destination) != summary[f'{source.stem.split("_")[0]}_sha256']:
                raise ValueError('Retained checkpoint copy differs')
        retained[step] = {**summary, 'policy': str((target / policy.name).relative_to(ROOT)),
                          'state': str((target / state.name).relative_to(ROOT)),
                          'label': f'seed 1; saved VOC {summary["voc_scale"]}'}
    reached = max(retained, default=0)
    if not {step for step in wanted if step <= reached}.issubset(retained):
        raise ValueError('A reached prescribed M0c checkpoint was not retained exactly')
    return [retained[step] for step in sorted(retained)], errors


def train(q, digest, sources):
    """Run one fresh training child under a lease and persist its complete checkpoint pairs."""
    if TRAIN.exists():
        raise RuntimeError('M0c training is single-use; no resume or retry')
    record = run_fields(ROOT, 'dynamic_feasibility', 'm0c_train', asdict(request(q)),
                        inputs=[QUEUE, ROOT / q['protocol'], ROOT / q['task_parquet'], handler.FEATURE / 'runtime.json'],
                        provenance='public_asset_diagnostic')
    record.update(identity=digest, admitted_sources=sources, pid=os.getpid(), seed=1,
                  diagnostic_only=True, challenge_episode_access=False,
                  retention_rule='All generated pairs remain; D1 pruning is never entered')
    with recorded(TRAIN, record, hash_artifacts=False):
        with GpuLease(ROOT, q['headroom_mib'], q['admission_allowance_mib'], stage='dynamic_feasibility_m0c_train') as lease:
            started = time.monotonic()
            try:
                handler.execute(training_command(q), TRAIN / 'training.log', environment(),
                                time.time() + q['train_seconds'], record, 'training')
            except RuntimeError as error:
                record['training_error'] = str(error)
            finally:
                record.update(gpu=lease.report(), gpu_seconds=time.monotonic() - started)
            lease.require_headroom()
        record['retained'], record['checkpoint_errors'] = retain_pairs(q)
        worker = TRAIN / 'worker.json'
        record['worker'] = json.loads(worker.read_text()) if worker.exists() else {'reason': 'crash'}
        record['progress'], record['compile_seconds'] = parse_progress((TRAIN / 'training.log').read_text())
        record['budget_overrun_seconds'] = max(0., record['gpu_seconds'] - q['train_seconds'])
        final = record['retained'][-1] if record['retained'] else None
        record['complete'] = bool(final and final['environment_steps'] == q['final_steps'] and final['voc_scale'] == 0.)
        record['completion_label'] = 'complete' if record['complete'] else 'incomplete'
    with score_output():
        m0b.ledger({**record, '_path': TRAIN / 'run.json'}, 'train')
    write_json(OUTPUT / 'train-finished.json', {'identity': digest, 'run_sha256': sha256_file(TRAIN / 'run.json'),
               'complete': record['complete'], 'completion_label': record['completion_label']})
    return record


def launch_train(q, digest):
    """Detach one owned training process and immediately report its PID and log."""
    path = OUTPUT / 'launch.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['identity'] != digest:
            raise ValueError('M0c launch identity changed')
        return saved
    if TRAIN.exists():
        raise RuntimeError('M0c training already exists; never launch again')
    command = [str(handler.PYTHON), '-B', '-m', MODULE, 'train', '--foreground']
    with (OUTPUT / 'train-launch.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {'identity': digest, 'pid': child.pid, 'argv': command,
              'log': str((OUTPUT / 'train-launch.log').relative_to(ROOT)),
              'finished': str((OUTPUT / 'train-finished.json').relative_to(ROOT)),
              'failure': 'terminal failed/interrupted trials/train/run.json or launcher exit without train-finished.json'}
    write_json(path, result)
    return result


@contextmanager
def score_output():
    """Redirect only M0b's measurement outputs in this process, leaving its sources immutable."""
    original = m0b.OUTPUT
    m0b.OUTPUT = OUTPUT
    try:
        yield
    finally:
        m0b.OUTPUT = original


def measure(phase, record_path):
    """Reuse M0b's exact CPU validity checks, metrics, final gate and motion audit."""
    with score_output():
        if phase == 'preflight':
            result = m0b.score_preflight()
            write_json(OUTPUT / 'scoring/preflight.json', result)
            return result
        record = json.loads(record_path.read_text())
        result = m0b.score_final(record)
    old = json.loads((m0b.OUTPUT / 'final.json').read_text())
    spread = {f'seed0_warm_start/{name}': row for name, row in old['rows'].items() if name.startswith('steps-')}
    spread.update({f'seed1_fresh/{trial["name"]}': result['rows'][trial['name']] for trial in record['trials']})
    times = {row['steps']: row['seconds'] for row in record['checkpoint_times']}
    for trial in record['trials']:
        result['rows'][trial['name']]['checkpoint_wall_seconds'] = times.get(trial['environment_steps'])
    result.update(seed=1, fresh_start=True, seed_spread=spread,
                  seed_spread_note='Two seeds with different starts; a spread, not a variance.')
    write_json(OUTPUT / 'final.json', result)
    path = OUTPUT / 'summary.md'
    text = path.read_text().replace('# M0b:', '# M0c:', 1)
    text += '\nSeed spread (different starts, not a variance):\n\n| Seed / start | T | Q | Position cm | ADD-AUC |\n| --- | ---: | ---: | ---: | ---: |\n'
    for name, row in result['seed_spread'].items():
        text += f'| {name} | ' + ' | '.join(str(row.get(key, 'INVALID')) for key in ('T', 'Q', 'mean_position_cm', 'add_auc')) + ' |\n'
    text += '\nCheckpoint wall seconds (includes setup, compilation and saves): ' + json.dumps(record['checkpoint_times']) + '.\n'
    text += 'Only the final/latest row is gated. Each retained policy has at most one evaluator invocation.\n'
    path.write_text(text)
    return result


def cpu_measure(phase, record_path):
    """Run measurement in the existing perception interpreter with accelerators hidden."""
    settings = json.loads((handler.FEATURE / 'runtime.json').read_text())
    command = [str(ROOT / settings['measurement_python']), '-B', '-m', MODULE,
               '--measure', phase, '--record', str(record_path)]
    env = {**environment(), 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'}
    with (record_path.parent / f'{phase}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((OUTPUT / ('scoring/preflight.json' if phase == 'preflight' else 'final.json')).read_text())


def evaluate_once(trial, env, deadline, record, record_path):
    """Reuse an existing attempt or invoke the shared evaluator once, with no crash retry."""
    checkpoint = ROOT / trial['checkpoint']
    trial['identity_passed'] = sha256_file(checkpoint) == trial['sha256']
    if not trial['identity_passed']:
        raise ValueError('Retained policy changed before scoring')
    directory = OUTPUT / f'scoring/runs/{trial["name"]}/attempt0'
    if directory.exists():
        attempt = json.loads((directory / 'run.json').read_text())
        if attempt['checkpoint_sha256'] != trial['sha256']:
            raise ValueError('Existing evaluator attempt belongs to another policy')
        for key in ('pid', 'evaluation_pid'):
            if attempt.get('status') != 'completed' and key in attempt and Path(f'/proc/{attempt[key]}').exists():
                raise RuntimeError('Existing evaluator attempt is still alive')
        for name, digest in attempt.get('artifacts', {}).items():
            if sha256_file(directory / name) != digest:
                raise ValueError('Existing evaluator export changed')
    elif time.time() < deadline:
        attempt = {'kind': 'dynamic_feasibility.m0c_evaluation', 'status': 'running',
                   'checkpoint_sha256': trial['sha256'], 'pid': os.getpid()}
        with recorded(directory, attempt):
            try:
                handler.execute(partial_eval.evaluation_command(checkpoint, directory), directory / 'evaluation.log',
                                env, deadline, attempt, 'evaluation')
            except RuntimeError as error:
                attempt['evaluation_error'] = str(error)
    else:
        trial['budget_exhausted'] = True
        return
    if (directory / 'trajectory.parquet').exists():
        trial['export_directory'] = str(directory.relative_to(ROOT))
    trial['attempts'] = [{'directory': str(directory.relative_to(ROOT)), 'error': attempt.get('evaluation_error'),
                          'seconds': attempt.get('evaluation_seconds')}]
    write_json(record_path, record)


def score(q, digest, sources):
    """Score retained rows in a resumable fixed loop and gate only the last complete policy."""
    finished = OUTPUT / 'score-finished.json'
    if finished.exists():
        saved = json.loads(finished.read_text())
        if saved['identity'] != digest or saved['final_sha256'] != sha256_file(OUTPUT / 'final.json'):
            raise ValueError('Completed M0c score changed')
        return json.loads((OUTPUT / 'final.json').read_text())
    training = m0b.previous(TRAIN, digest)
    if not training:
        raise RuntimeError('M0c training has not run')
    if training['status'] != 'completed' and any(Path(f'/proc/{training[key]}').exists()
            for key in ('pid', 'training_pid') if key in training):
        raise RuntimeError('An interrupted training operation still has a live owner or child')
    if 'retained' not in training:
        pairs, errors = retain_pairs(q)
        training.update(retained=pairs, checkpoint_errors=errors, complete=False,
                        completion_label='incomplete; recovered after interruption')
        write_json(TRAIN / 'recovery.json', {key: value for key, value in training.items() if key != '_path'})
    if not training['retained']:
        raise RuntimeError('No complete finite M0c pair exists to score')
    steps = [pair['environment_steps'] for pair in training['retained']]
    expected = sorted({step for step in q['retain_steps'] if step <= max(steps)} | {max(steps)})
    if steps != expected:
        raise ValueError('Retained policies do not match the prescribed reached rows and final/latest pair')
    for pair in training['retained']:
        summary = m0b.cpu_inspect(ROOT / pair['policy'], ROOT / pair['state'])
        if any(summary[key] != pair[key] for key in ('policy_sha256', 'state_sha256', 'environment_steps')):
            raise ValueError('Retained pair changed before scoring')
    invocations = OUTPUT / 'scoring/invocations'
    for path in invocations.glob('*/run.json'):
        old = json.loads(path.read_text())
        if old['status'] == 'running' and Path(f'/proc/{old["pid"]}').exists():
            raise RuntimeError('A scoring invocation is still alive')
    directory = invocations / str(len(list(invocations.glob('*'))))
    record = run_fields(ROOT, 'dynamic_feasibility', 'm0c_score', {'seed': 1},
                        inputs=[QUEUE, TRAIN / 'run.json'], provenance='public_asset_diagnostic')
    worker = TRAIN / 'worker.json'
    retained_steps = {pair['environment_steps'] for pair in training['retained']}
    checkpoint_times = json.loads(worker.read_text()).get('checkpoints', []) if worker.exists() else []
    record.update(identity=digest, admitted_sources=sources, pid=os.getpid(),
                  training={key: training[key] for key in ('complete', 'completion_label')},
                  recipe={'ratio': q['ratio'], 'capacity': q['capacity'], 'lr_horizon': q['lr_horizon']},
                  checkpoint_times=[row for row in checkpoint_times if row['steps'] in retained_steps],
                  trials=[{**pair, 'name': f'steps-{pair["environment_steps"]}', 'source': pair['policy'],
                           'checkpoint': pair['policy'], 'sha256': pair['policy_sha256'], 'attempts': []}
                          for pair in training['retained']])
    with recorded(directory, record, hash_artifacts=False):
        pre = cpu_measure('preflight', directory / 'run.json')
        if not all(row['passed'] for row in pre['checks'].values()):
            raise ValueError('M0c pre-evaluation validity failure')
        with GpuLease(ROOT, q['headroom_mib'], 4096, stage='dynamic_feasibility_m0c_score') as lease:
            started = time.monotonic()
            deadline = time.time() + q['score_seconds']
            for trial in record['trials']:
                evaluate_once(trial, environment(), deadline, record, directory / 'run.json')
                lease.require_headroom()
            record.update(gpu=lease.report(), gpu_seconds=time.monotonic() - started)
        write_json(directory / 'run.json', record)
        result = cpu_measure('final', directory / 'run.json')
    with score_output():
        m0b.ledger({**record, '_path': directory / 'run.json'}, 'score')
    write_json(finished, {'identity': digest, 'final_sha256': sha256_file(OUTPUT / 'final.json'),
                         'run': str((directory / 'run.json').relative_to(ROOT))})
    return result


def worker_main(arguments):
    """Run upstream math unchanged with fresh progress counters and cooperative stopping."""
    from ._m0b_worker import BudgetStop
    origin = time.monotonic()
    deadline = float(os.environ['DYNAMIC_FEASIBILITY_DEADLINE'])
    last, checkpoints, reason = None, [], 'completed'
    calls = 0

    def setup_guard(frame, event, arg):
        nonlocal calls
        calls += 1
        if calls % 4096 == 0 and last is None and time.time() >= deadline:
            raise BudgetStop('deadline_during_setup')

    sys.setprofile(setup_guard)
    try:
        from flash_chord.training import progress
        from flash_chord.training.flash_sac import runner
        original = progress.ConsoleTrainingProgress
        save = runner.save_learner_checkpoint

        class Clock(original):
            def start(self, steps):
                nonlocal last
                if steps != 0:
                    raise ValueError('M0c must start from zero')
                last = {'steps': 0, 'updates': 0, 'seconds': time.monotonic() - origin}
                print(f'M0B t={last["seconds"]:.9f} updates=0 replay=0 credit=0 ', end='')
                super().start(steps)
                self.last_report = time.monotonic()
                if time.time() >= deadline:
                    raise BudgetStop('deadline')

            def update(self, steps):
                nonlocal last
                stopping = time.time() >= deadline
                if time.monotonic() - self.last_report >= 1 or stopping or steps == self.total_environment_steps:
                    import jax
                    learner = sys._getframe(1).f_locals['learner']
                    updates, size, credit = jax.device_get((learner.global_update_step, learner.replay.size, learner.update_credit))
                    last = {'steps': steps, 'updates': int(updates), 'seconds': time.monotonic() - origin}
                    print(f'M0B t={last["seconds"]:.9f} updates={int(updates)} replay={int(size)} credit={float(credit):.9f} ', end='')
                    self._next_percent = 0
                    super().update(steps)
                    self.last_report = time.monotonic()
                if stopping:
                    raise BudgetStop('deadline')

        def timed_save(*args, **kwargs):
            paths = save(*args, **kwargs)
            checkpoints.append({'steps': kwargs['metadata']['environment_steps'],
                                'seconds': time.monotonic() - origin})
            write_json(TRAIN / 'worker.json', {'reason': 'running', 'checkpoints': checkpoints, 'last_progress': last})
            return paths

        progress.ConsoleTrainingProgress = Clock
        runner.save_learner_checkpoint = timed_save
        script, *overrides = arguments
        sys.argv = [script, *overrides]
        runpy.run_path(script, run_name='__main__')
    except BudgetStop as error:
        reason = str(error)
    except BaseException as error:
        reason = f'error:{type(error).__name__}'
        raise
    finally:
        sys.setprofile(None)
        write_json(TRAIN / 'worker.json', {'reason': reason, 'last_progress': last,
                   'checkpoints': checkpoints, 'wall_seconds': time.monotonic() - origin})
        print(f'M0C_STOP {reason}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', nargs='?', choices=('train', 'score'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check-m0b-identity', action='store_true')
    parser.add_argument('--foreground', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker', nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    parser.add_argument('--measure', choices=('preflight', 'final'), help=argparse.SUPPRESS)
    parser.add_argument('--record', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        runtime('train')
        environment()
        worker_main(args.worker)
        return
    if args.check_m0b_identity:
        print(json.dumps(check_m0b_identity(), indent=2))
        return
    if args.measure:
        runtime('score', measurement=True)
        queue()
        environment()
        print(json.dumps(measure(args.measure, args.record)['verdict'] if args.measure == 'final'
                         else measure('preflight', args.record)['checks']))
        return
    if not args.operation:
        parser.error('Choose train or score')
    q = queue()
    if args.dry_run:
        print(json.dumps(dry_run(args.operation, q), indent=2))
        return
    runtime(args.operation)
    environment()
    check_m0b_identity()
    digest, sources = identity(q)
    if args.operation == 'train':
        result = train(q, digest, sources) if args.foreground else launch_train(q, digest)
    else:
        result = score(q, digest, sources)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
