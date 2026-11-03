"""Run the frozen two-object queue once, resuming completed identities."""

from dataclasses import asdict
import hashlib
import json
import os
import time

from track3.paths import output_path, project_path
from track3.provenance import read_run, run_fields, write_json
from track3.runs import ROOT, authorize_episode, new_run_id, read_json, recorded, source_hashes

from .handler import FEATURE, PROTOCOL, REUSED, cpu_runtime, stage_a
from .request import StageARequest


def load_ledger(path):
    rows = {}
    if path.is_file():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                continue
            if row['status'] == 'completed':
                rows[row['identity']] = row
    return rows


def append_ledger(path, row):
    needs_newline = path.is_file() and path.stat().st_size and not path.read_bytes().endswith(b'\n')
    with path.open('a', encoding='utf-8') as stream:
        if needs_newline:
            stream.write('\n')
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()


def summary(rows, directory):
    lines = ['# F1 Stage A — video-only feasibility', '',
             'Frozen protocol: docs/chain/mesh-fusion-protocol.md (277c7df). CPU only; no scan or CD-O.', '',
             '| Object / public episode | Pre-contact interval | Candidates | Accepted | Angle deg | Additional 3 mm support % | Verdict |',
             '| --- | --- | --- | --- | ---: | ---: | --- |']
    for row in sorted(rows, key=lambda r: (r['result']['verdict'] != 'PASS', r['episode'])):
        result = row['result']
        lines.append(f"| {result['object']} / {row['episode']} | [0,{row['contact']}) | {row['candidates']} | "
                     f"{result['accepted_frames']} | {result['angular_span_deg']:.6f} | "
                     f"{result['additional_supported_surface_percent']:.6f} | {result['verdict']} |")
    lines += ['', '**Frozen kill rule:** fewer than 3 accepted frames, OR angular span <10 degrees AND additional supported surface <10%.',
              'Frame acceptance: >=30 PnP inliers, median reprojection <=1 px, independent background disagreement <=3 mm.',
              'The independent check uses >=10 unused descriptor correspondences and median 3D Euclidean error, without error trimming.',
              'Frame 0 is the identity reference. Contact boundaries are conservative annotations with visible-contact uncertainty recorded per run.',
              'Additional support is exact voxel novelty, not a denoised estimate of newly exposed shape.', '']
    for row in rows:
        folder = row['run'].rsplit('/', 1)[0]
        relative = os.path.relpath(ROOT / folder, directory)
        lines += [f"- {row['result']['object']}: [{row['run']}]({relative}/run.json); "
                  f"[acceptance table]({relative}/summary.md), [contact sheet]({relative}/contact-refined.png), "
                  f"[camera centres]({relative}/camera-centres.png)."]
    if rows and all(row['result']['verdict'] == 'KILL' for row in rows):
        lines += ['', '**F1 ends under this protocol. Both objects keep v4; no Stage B run is authorized by these results.**']
    else:
        lines += ['', 'Stage B remains unrun; any passing object needs one written recipe committed by the lead before scoring.']
    return '\n'.join(lines) + '\n'


def run_queue(queue='track3/features/mesh_fusion/queue.json'):
    runtime = cpu_runtime()
    queue_path = project_path(ROOT, queue)
    spec = read_json(queue_path)
    if spec['protocol'] != PROTOCOL:
        raise ValueError('The queue must name the committed F1 protocol')
    requests = [StageARequest(**trial) for trial in spec['trials']]
    for request in requests:
        authorize_episode(runtime, request, 'F1 Stage A')
    sources = source_hashes(FEATURE, [*REUSED, PROTOCOL, FEATURE / 'runtime.json'])
    record = run_fields(ROOT, 'mesh_fusion', 'queue', spec,
                        inputs=[queue_path, PROTOCOL, FEATURE / 'runtime.json'])
    record.update(run_id=new_run_id(), gpu=None, trials=[], source_hashes=sources)
    base = output_path(ROOT, 'outputs/mesh_fusion')
    directory = base / 'queues' / record['run_id']
    ledger = base / 'ledger.jsonl'
    previous = load_ledger(ledger)
    started, rows = time.perf_counter(), []
    with recorded(directory, record):
        for request in requests:
            if time.perf_counter() - started > spec['budget_seconds']:
                raise TimeoutError('F1 queue time budget exhausted between objects')
            identity = hashlib.sha256(json.dumps(dict(request=asdict(request), sources=sources), sort_keys=True).encode()).hexdigest()
            if identity in previous:
                row = previous[identity]
                read_run(ROOT, project_path(ROOT, row['run']), allowed=('video_only',), kind='mesh_fusion.stage_a')
                print(f"resume completed public{request.episode_index}: {row['run']}", flush=True)
            else:
                run_path = stage_a(request, identity)
                completed = read_json(run_path)
                row = dict(identity=identity, status='completed', episode=request.episode_index,
                           contact=request.first_contact_frame, candidates=request.candidate_frames,
                           run=run_path.relative_to(ROOT).as_posix(), result=completed['result'])
                append_ledger(ledger, row)
            rows.append(row)
            record['trials'].append(row)
            write_json(directory / 'run.json', record)
        (directory / 'summary.md').write_text(summary(rows, directory), encoding='utf-8')
        (base / 'summary.md').write_text(summary(rows, base), encoding='utf-8')
    print(base / 'summary.md', flush=True)
    return directory / 'run.json'
