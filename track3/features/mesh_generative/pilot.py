"""Gated and resumable orchestration for the frozen three-object G1 pilot."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from track3.gpu import GpuLease
from track3.paths import PROJECT as ROOT, output_path, project_path
from track3.provenance import input_file, run_fields, sha256_file, write_json
from track3.runs import CACHE_FOLDERS, authorize_episode, now_utc, runtime_flag

FEATURE = Path(__file__).parent
OUT = 'outputs/mesh_generative/g1-pilot'
WEIGHTS = '.runtime/hunyuan-mesh/weights-xet'
EXPECTED = {(12, 'plastic_dish_rack'), (21, 'water_pitcher'), (42, 'white_pot')}


@dataclass(frozen=True)
class PilotRequest:
    queue: str = 'track3/features/mesh_generative/queue.json'

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(project_path(ROOT, path).read_text()))


def authorize():
    """Check the runtime and redirect caches before any numerical/model imports."""
    runtime = runtime_flag(FEATURE, 'pilot_authorized', 'Frozen G1 pilot')
    if Path(sys.executable).resolve() != project_path(ROOT, runtime['python']):
        raise RuntimeError('Run the pilot handler with .runtime/linux-cuda/bin/python')
    for key, value in CACHE_FOLDERS.items():
        folder = '.runtime/tmp/g1' if key in ('TMP','TEMP','TMPDIR') else value
        path = output_path(ROOT, folder)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    tempfile.tempdir = os.environ['TMPDIR']
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                      PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'upstream/v2d_submission_kit/v2d_submission_kit'))
    return runtime


def generate(image, mesh):
    """Hold the project GPU lease for exactly one isolated generation subprocess."""
    if mesh.exists() and mesh.with_suffix('.json').exists():
        return json.loads(mesh.with_suffix('.json').read_text())
    started = mesh.with_suffix('.started.json')
    if started.exists():
        raise RuntimeError(f'Unfinished generation; no automatic re-roll allowed: {started}')
    with GpuLease(ROOT, 1024, 10000, stage='mesh_generative') as lease:
        write_json(started, dict(image=str(image), started_utc=now_utc()))
        command = [str(ROOT/'.runtime/hunyuan-mesh/venv/bin/python'),
                   str(FEATURE/'worker.py'), '--image', str(image), '--output', str(mesh),
                   '--weights', str(ROOT/WEIGHTS)]
        with mesh.with_suffix('.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        telemetry = lease.report()
    report = json.loads(mesh.with_suffix('.json').read_text())
    report['gpu_lease'] = telemetry
    write_json(mesh.with_suffix('.json'), report)
    return report


def smoke():
    """Generate once from the pinned upstream demo image, outside the dataset."""
    authorize()
    directory = output_path(ROOT, '.runtime/tmp/g1/smoke')
    directory.mkdir(parents=True, exist_ok=True)
    revision = (ROOT/'.runtime/hunyuan-mesh/code-revision.txt').read_text().strip()
    image = ROOT/f'.runtime/hunyuan-mesh/source/Hunyuan3D-2-{revision}/assets/demo.png'
    record = run_fields(ROOT, 'mesh_generative', 'smoke', {}, inputs=[image, ROOT/'.runtime/hunyuan-mesh/weights-selected.json',
                    ROOT/'.runtime/hunyuan-mesh/requirements.lock.txt',
                    ROOT/f'.runtime/hunyuan-mesh/{revision}.zip',
                    *sorted((ROOT/WEIGHTS/'hunyuan3d-dit-v2-mini').glob('*'))])
    write_json(directory/'run.json', record)
    try:
        record.update(generation=generate(image, directory/'demo.ply'), status='completed')
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['finished_utc'] = now_utc()
        record['artifacts'] = {p.name: sha256_file(p) for p in directory.iterdir()
                               if p.is_file() and p.name != 'run.json'}
        write_json(directory/'run.json', record)
    return directory/'run.json'


def prepare_crop(job, directory):
    """Use the recorded frame-zero RGB overlay and its exact assignment label mask."""
    import cv2
    import numpy as np
    from ._geometry import masked_square

    parent = project_path(ROOT, job['measurement'])
    producer = json.loads(parent.read_text())
    if (producer['status'] != 'completed' or producer['provenance'] != 'video_only'
            or producer['episode_index'] != job['episode'] or producer['split'] != 'public'):
        raise ValueError('Crop requires a completed matching video-only measurement')
    if producer['frames'][0]['frame_index'] != 0:
        raise ValueError('Pilot crops require the recorded frame-zero mask')
    video = next(entry for entry in producer['inputs']
                 if '/observation.images.ego_cam_a/' in entry['path']
                 and entry['path'].endswith(f"episode_{job['episode']:06d}.mp4"))
    assignment_path = ROOT/'track3/features/mesh_extents/assignments.json'
    assignment = json.loads(assignment_path.read_text())['episodes'][f"public:{job['episode']}"]
    ids = assignment['objects'][job['object']]
    if producer['request']['objects'][job['object']] != ids:
        raise ValueError('Recorded cluster assignment changed')
    names = ['labels_000000.npz','overlay_000000.png']
    inputs = [input_file(ROOT, parent), input_file(ROOT, assignment_path)]
    for name in names:
        entry = input_file(ROOT, parent.parent/name)
        if entry['sha256'] != producer['artifacts'][name]:
            raise ValueError(f'Recorded mask/image hash mismatch: {name}')
        inputs.append(entry)
    overlay = cv2.imread(str(parent.parent/names[1]))
    rgb = cv2.cvtColor(overlay[:, :overlay.shape[1]//2], cv2.COLOR_BGR2RGB)
    with np.load(parent.parent/names[0], allow_pickle=False) as arrays:
        mask = np.isin(arrays['label'], [i+1 for i in ids])
    crop, bounds = masked_square(rgb, mask)
    image = directory/'crop.png'
    if not image.exists():
        cv2.imwrite(str(image), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    else:
        stored = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        if not np.array_equal(stored, crop):
            raise ValueError('Frozen crop differs; refusing a crop change')
    inputs.append(input_file(ROOT, image))
    return inputs, dict(source_video=video, cluster_ids=ids, mask_pixels=int(mask.sum()), bounds=bounds,
                        margin='10 percent of the maximum mask extent on each side',
                        rgb='left half of lossless recorded overlay: decoded frame-0 camera-A, depth remap')


def score(job, directory):
    """Score the single shape against the incumbent object's original scan seeds on CPU."""
    score_path = directory/'scores.json'
    if not score_path.exists():
        command = [str(ROOT/'.runtime/kit-env/bin/python'), '-m', 'track3.features.mesh_generative.score',
                   '--mesh', str(directory/'raw.ply'), '--object', job['object'], '--output', str(score_path)]
        with (directory/'score.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(score_path.read_text())


def pilot(request):
    """Run the fixed queue once, resume completed shapes, and stop at the pilot gate."""
    runtime = authorize()
    queue_path = project_path(ROOT, request.queue)
    jobs = json.loads(queue_path.read_text())['objects']
    if len(jobs) != 3 or {(j['episode'], j['object']) for j in jobs} != EXPECTED:
        raise PermissionError('Only the frozen three-object pilot is authorized')
    for job in jobs:
        episode_request = type('Episode', (), dict(source_split='public',episode_index=job['episode']))()
        authorize_episode(runtime, episode_request, 'G1 generation')
    smoke_path = ROOT/'.runtime/tmp/g1/smoke/run.json'
    if json.loads(smoke_path.read_text())['status'] != 'completed':
        raise RuntimeError('Smoke generation must pass first')
    directory = output_path(ROOT, OUT)
    directory.mkdir(parents=True, exist_ok=True)
    install = ROOT/'.runtime/hunyuan-mesh'
    revision = (install/'code-revision.txt').read_text().strip()
    inputs = [queue_path, FEATURE/'runtime.json', 'docs/chain/mesh-generative-protocol.md',
              install/f'{revision}.zip', install/'weights-selected.json', install/'requirements.lock.txt',
              *sorted((ROOT/WEIGHTS/'hunyuan3d-dit-v2-mini').glob('*'))]
    frozen_crops = {}
    for job in jobs:
        item = directory/job['object']
        item.mkdir(parents=True, exist_ok=True)
        crop_inputs, crop = prepare_crop(job, item)
        frozen_crops[job['object']] = (crop_inputs, crop)
        inputs.extend(ROOT/entry['path'] for entry in crop_inputs)
    record = run_fields(ROOT, 'mesh_generative', 'pilot', asdict(request), inputs=inputs)
    record['frozen_crops'] = {name: crop for name, (_, crop) in frozen_crops.items()}
    record['sources'] += [input_file(ROOT,p) for p in
        ('scripts/track3/local_cdo.py','track3/features/mesh_eval/pairs.py',
         'track3/features/mesh_eval/_metrics.py',
         'upstream/v2d_submission_kit/v2d_submission_kit/v2dlb/mesh_budget.py',
         'upstream/v2d_submission_kit/v2d_submission_kit/metric_code/track_3/CD-O.py')]
    identity = sha256_file(queue_path)
    previous_path = directory/'run.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if any(previous[key] != record[key] for key in ('inputs','sources','request')):
            raise ValueError('Frozen run identity changed; do not regenerate')
        record = previous
        if record['status'] == 'completed':
            return previous_path
    record.update(status='running', queue_identity=identity, started_utc=record.get('started_utc',now_utc()))
    write_json(previous_path, record)
    try:
        all_scores, objects = [], []
        for job in jobs:
            item = directory/job['object']
            item.mkdir(parents=True, exist_ok=True)
            inputs, crop = frozen_crops[job['object']]
            generation = generate(item/'crop.png', item/'raw.ply')
            measured = score(job, item)
            all_scores.extend(measured['rows'])
            objects.append(dict(**job, inputs=inputs+measured['inputs'], crop=crop, generation=generation))
            record['objects'] = objects
            write_json(previous_path, record)
            ledger = directory/'ledger.jsonl'
            completed = {json.loads(line)['object'] for line in ledger.read_text().splitlines()} if ledger.exists() else set()
            if job['object'] not in completed:
                with ledger.open('a') as stream:
                    stream.write(json.dumps(dict(**job, status='completed', mesh_sha256=sha256_file(item/'raw.ply'),
                                                 crop_sha256=sha256_file(item/'crop.png'),generation=generation,
                                                 scores=measured['rows']))+'\n')
                    stream.flush(); os.fsync(stream.fileno())
        from ._geometry import pilot_metrics
        from .report import publish
        summary = pilot_metrics(all_scores)
        write_json(directory/'summary.json', summary)
        publish(directory, summary, objects)
        record.update(status='completed', verdict=summary['verdict'], finished_utc=now_utc(),
                      provenance='public_asset_diagnostic', generation_provenance='video_only')
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['finished_utc'] = now_utc()
        record['artifacts'] = {p.relative_to(directory).as_posix(): sha256_file(p)
                               for p in directory.rglob('*') if p.is_file() and p.name != 'run.json'}
        write_json(previous_path, record)
    return previous_path
