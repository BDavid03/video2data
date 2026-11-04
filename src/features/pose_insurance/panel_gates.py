"""Freeze and score the protocol's CPU-only, train-label Stage C panel.

Run: CUDA_VISIBLE_DEVICES='' PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m
src.features.pose_insurance.panel_gates
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from src.provenance import input_file, sha256_file, write_json
from src.runs import ROOT, artifact, authorize_episode, checked_entry, producer, read_json, runtime_flag

FEATURE = Path(__file__).parent
OUTPUT = ROOT / 'data/outputs/pose_insurance/frame0-panel/stage-c'
PANEL = OUTPUT.parent
PROTOCOL = 'docs/chain/frame0-panel-protocol.md'
COMMIT = 'cca187a'
OLD_PROTOCOL = 'd3890b96b7b5881558795ccf7395facc4ffab1400151f7360bb7d167e153368a'
NEW_PROTOCOL = 'b565cb3d63e1600b440e2a60b2e6a17f99b2055bd1bec34cb529af9350233605'


def stamp() -> str:
    """Return UTC wall-clock evidence for construction and label-access ordering."""
    return datetime.now(timezone.utc).isoformat()


def binding(value: dict) -> str:
    """Hash a canonical JSON request for exact trial resumption."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def git(*args: str) -> bytes:
    """Read repository evidence without mutating its state."""
    return subprocess.check_output(['git', *args], cwd=ROOT)


def preflight() -> dict:
    """Enforce every seal hash with only the lead's exact protocol-history exception."""
    seal = read_json(PANEL / 'label-free-seal.json')
    checks = []
    for group in ('artifacts', 'sources'):
        for entry in seal[group]:
            actual = sha256_file(ROOT / entry['path'])
            checks.append(dict(group=group, **entry, actual_sha256=actual, passed=actual == entry['sha256']))
    for group in ('method', 'queue'):
        entry = seal[group]
        actual = sha256_file(ROOT / entry['path'])
        checks.append(dict(group=group, **entry, actual_sha256=actual, passed=actual == entry['sha256']))
    old = hashlib.sha256(git('show', f'{COMMIT}^:{PROTOCOL}')).hexdigest()
    committed = hashlib.sha256(git('show', f'{COMMIT}:{PROTOCOL}')).hexdigest()
    actual = sha256_file(ROOT / PROTOCOL)
    exception = dict(path=PROTOCOL, reason='Lead-authorized Amendment 1; preserve the pre-amendment seal.',
                     sealed_sha256=seal['protocol']['sha256'], parent_commit_sha256=old,
                     amendment_commit_sha256=committed, disk_sha256=actual,
                     parent_matches_seal=(seal['protocol']['path'] == PROTOCOL
                                          and old == seal['protocol']['sha256'] == OLD_PROTOCOL),
                     disk_matches_amendment=actual == committed == NEW_PROTOCOL)
    commit = git('rev-parse', f'{COMMIT}^{{commit}}').decode().strip()
    head = git('rev-parse', 'HEAD').decode().strip()
    ancestor = dict(required_commit=commit, head=head,
                    command='git log --format=%H HEAD',
                    passed=commit in git('log', '--format=%H', 'HEAD').decode().splitlines())
    status = git('status', '--porcelain', '--', PROTOCOL).decode()
    result = dict(checked_utc=stamp(), seal=input_file(ROOT, PANEL / 'label-free-seal.json'),
                  artifact_count=len(seal['artifacts']), strict_hash_count=len(checks), checks=checks,
                  protocol_exception=exception, ancestor_check=ancestor,
                  protocol_uncommitted_status=status, protocol_clean=not status)
    result['passed'] = (all(row['passed'] for row in checks) and ancestor['passed'] and not status
                        and exception['parent_matches_seal'] and exception['disk_matches_amendment'])
    if not result['passed']:
        write_json(OUTPUT / f'preflight-refusal-{time.time_ns()}.json', result)
        raise PermissionError('Stage C preflight refused; see preflight-refusal artifact')
    return result


def read_inputs(record: dict) -> tuple:
    """Read only label-free T1 constructions, planes and frozen registration artifacts."""
    import numpy as np
    from ._panel_gates import EPISODES, admitted
    from ._stereo_placement import start_poses
    queue_path = FEATURE / 'heading_queue.json'
    queue = read_json(queue_path)
    record['inputs'].append(input_file(ROOT, queue_path))
    stage_a_path, stage_b_path = PANEL / 'stage-a.json', PANEL / 'stage-b.json'
    stage_a = {row['episode']: row for row in read_json(stage_a_path)['episodes']}
    stage_b = read_json(stage_b_path)['predictions']
    record['inputs'].extend(input_file(ROOT, p) for p in (stage_a_path, stage_b_path))
    data = {}
    for ep in EPISODES:
        request = SimpleNamespace(source_split='public', episode_index=ep)
        parent = producer(queue['builds'][str(ep)], 'pose_insurance.stereo_placement_build', request,
                          allowed=('video_only',))
        record['inputs'].append(input_file(ROOT, parent[0]))
        plan = read_json(artifact(parent, 'construction.json', record))
        if plan['episode'] != ep or not plan['unchanged_fields']:
            raise ValueError('T1 build does not match the immutable train construction')
        scene = {key: np.asarray(value) for key, value in plan['scenes']['T1'].items()}
        start = start_poses(scene)
        # Plane records are video-only, admitted by the original T1 build.
        plane_entry = next(entry for entry in parent[1]['inputs']
                           if entry['path'].startswith('data/outputs/mesh_extents/public/episode_')
                           and entry['path'].endswith('/run.json'))
        plane_path = checked_entry(plane_entry, record)
        plane = producer(plane_path, 'mesh_extents.measure', request, allowed=('video_only',))
        normal = np.asarray(plane[1]['frames'][0]['plane']['normal'])
        registrations = []
        if ep in (0, 26, 31):
            registrations = sorted([r for r in stage_b if r['episode'] == ep], key=lambda r: r['slot'])
            for row in registrations:
                if row['admitted'] != admitted(row):
                    raise ValueError('Stage B admission differs from the fixed panel predicate')
        elif ep in (18, 23):
            for slot, entry in enumerate(queue['h2s']['registrations'][str(ep)]):
                path = checked_entry(entry, record)
                parent_reg = producer(path, 'pose_insurance.registration_trial', request,
                                      allowed=('public_asset_diagnostic',))
                row = read_json(artifact(parent_reg, 'prediction.json', record))
                if (row['episode'], row['slot'], row['name']) != (ep, slot, plan['names'][slot]):
                    raise ValueError('Historical registration identity differs from train slots')
                arrays_path = artifact(parent_reg, 'registrations.npz', record)
                with np.load(arrays_path, allow_pickle=False) as arrays:
                    best = int(arrays['search_0_losses_m'].argmin())
                    if best != row['best_index'] or not np.array_equal(
                            arrays['search_0_rotations'][best], row['rotation']):
                        raise ValueError('Selected historical rotation differs from search zero')
                    row['rival_rotation'] = (None if row.get('rival_index') is None else
                                             arrays['search_0_rotations'][row['rival_index']].tolist())
                row['admitted'] = admitted(row)
                registrations.append(row)
        if registrations and [r['slot'] for r in registrations] != [0, 1]:
            raise ValueError('Complete registration pair or explicit failed slots required')
        data[ep] = dict(start=start, scene=scene, names=plan['names'], normal=normal,
                        stage_a=stage_a.get(ep, {}), registrations=registrations)
    return data, queue


def freeze(data: dict, sources: list) -> dict:
    """Save every arm and distinct applied flip combination before opening labels."""
    from ._panel_gates import ARMS, EPISODES, construction
    payload = dict(schema_version=1, protocol=input_file(ROOT, PROTOCOL), sources=sources,
                   episodes=list(EPISODES), surface_seeds=[0, 1, 2], arms={}, queue=[])
    for arm in ARMS:
        payload['arms'][arm] = {}
        for ep in EPISODES:
            item = data[ep]
            row = construction(item['start'], ep, arm, item['stage_a'],
                               item['registrations'], item['normal'], item['scene']['inertia_diagonal_kg_m2'])
            row['names'] = item['names']
            for variant in row['variants']:
                variant['pose_key'] = binding(dict(episode=ep, poses=variant['poses']))
                for seed in payload['surface_seeds']:
                    trial = dict(episode=ep, seed=seed, pose_key=variant['pose_key'], poses=variant['poses'])
                    if not any(old['pose_key'] == trial['pose_key'] and old['seed'] == seed
                               for old in payload['queue']):
                        payload['queue'].append(trial)
            payload['arms'][arm][str(ep)] = row
    for ep in (26, 31):
        if payload['arms']['P'][str(ep)] != payload['arms']['P-free'][str(ep)]:
            raise ValueError('Amendment 1 requires P and P-free coincidence on 26/31')
    path = OUTPUT / 'candidates.json'
    if path.exists():
        if read_json(path) != payload:
            raise PermissionError('Frozen candidates differ; arm/source revisions after scoring are forbidden')
    else:
        write_json(path, payload)
        with (OUTPUT / 'candidates.sha256').open('x') as stream:
            stream.write(sha256_file(path) + '  candidates.json\n')
    expected = (OUTPUT / 'candidates.sha256').read_text().split()[0]
    if sha256_file(path) != expected:
        raise PermissionError('Frozen candidate hash differs')
    return payload


def json_ready(value):
    """Serialize nonfinite diagnostics as missing values after failed clause evaluation."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def score_queue(candidates: dict, data: dict, record: dict) -> tuple:
    """Resume exact CPU trials, appending hashes and metrics to one fixed ledger."""
    import numpy as np
    import trimesh
    from ._panel_gates import EPISODES, frame_errors
    from .heading_loop import append
    from .placement_diagnostic import baseline_module, reference
    from .stereo_placement_loop import scoring_inputs
    baseline = baseline_module()
    kit = baseline.load_kit()
    if (not record['preflight']['passed'] or read_json(OUTPUT / 'candidates.json') != candidates
            or sha256_file(OUTPUT / 'candidates.json') != record['candidates']['sha256']):
        raise PermissionError('Candidate freeze and passing preflight required before labels')
    record['first_label_access_utc'] = stamp()
    ledger = OUTPUT / 'ledger.jsonl'
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    refs, meshes, vertices, inputs = {}, {}, {}, {}
    for ep in EPISODES:
        paths = scoring_inputs(ep, data[ep]['names'])
        inputs[ep] = [input_file(ROOT, p) for p in paths]
        record['inputs'].extend(inputs[ep])
        refs[ep] = reference(ep)
    scores, executed, resumed = {}, 0, 0
    for trial in candidates['queue']:
        ep, seed = trial['episode'], trial['seed']
        identity = binding(dict(trial=trial, inputs=inputs[ep], sources=record['sources'],
                                candidates_sha256=record['candidates']['sha256']))
        old = next((r for r in reversed(previous) if r['identity'] == identity), None)
        path = OUTPUT / 'trials' / f'{identity}.json'
        if old:
            if sha256_file(path) != old['sha256']:
                raise ValueError('Completed trial hash differs from its ledger')
            result = read_json(path)
            if result['identity'] != identity or result['status'] != 'completed':
                raise ValueError('Completed trial identity/status differs')
            resumed += 1
        else:
            tick = time.monotonic()
            samples = []
            for name in data[ep]['names']:
                if name not in meshes:
                    scan = Path(baseline.ROOT) / f'mesh/{name}/{name}.glb'
                    if not scan.exists():
                        scan = scan.with_name(f'{name}_visual.glb')
                    meshes[name] = trimesh.load(scan, force='mesh')
                if (name, seed) not in vertices:
                    vertices[name, seed] = np.asarray(
                        trimesh.sample.sample_surface(meshes[name], 1000, seed=seed)[0], dtype=np.float64)
                samples.append(vertices[name, seed])
            ref, pose = refs[ep], np.asarray(trial['poses'])
            metrics = baseline.score(kit, np.repeat(pose[None, None], len(ref), axis=0), ref, np.stack(samples))
            result = dict(kind='pose_insurance.panel_gates_trial', status='completed', identity=identity,
                          episode=ep, seed=seed, pose_key=trial['pose_key'], metrics=metrics,
                          frame0=frame_errors(pose, pose, ref[0, 0]), frames=len(ref),
                          inputs=inputs[ep], sources=record['sources'],
                          candidates=record['candidates'], wall_seconds=time.monotonic()-tick)
            result = json_ready(result)
            write_json(path, result)
            append(ledger, dict(identity=identity, path=str(path.relative_to(ROOT)),
                                sha256=sha256_file(path), episode=ep, seed=seed, metrics=metrics))
            executed += 1
        scores[trial['pose_key'], seed] = result
        print(json.dumps(dict(episode=ep, seed=seed, completed=len(scores),
                              total=len(candidates['queue']), resumed=bool(old))), flush=True)
    record['trials'] = dict(executed=executed, resumed=resumed, total=len(scores))
    return scores, refs


def summarize(candidates: dict, scores: dict, data: dict, refs: dict, record: dict) -> dict:
    """Assemble seed reports, frame-zero diagnostics and all immutable gate clauses."""
    import numpy as np
    from ._panel_gates import (ARMS, EPISODES, METRICS, assess, bias_reading,
                               frame_errors, limits, mean_metrics)
    results, means, gates = {}, {}, {}
    for seed in candidates['surface_seeds']:
        results[str(seed)], means[str(seed)], gates[str(seed)] = {}, {}, {}
        for arm in ARMS:
            rows = {}
            for ep in EPISODES:
                key = str(ep)
                candidate = candidates['arms'][arm][key]
                variants = [scores[v['pose_key'], seed] for v in candidate['variants']]
                row = dict(variants[0])
                row['flip_stress'] = variants
                row['limits'] = {kind: limits(row['frame0'], candidate['rotation_admitted'], kind)
                                 for kind in ('TW', 'AN')}
                row['positions_baseline'] = candidate['positions_baseline']
                row['unchanged_from_T1'] = bool(np.array_equal(
                    np.asarray(candidate['variants'][0]['poses']), data[ep]['start']))
                rows[key] = row
            results[str(seed)][arm] = rows
            means[str(seed)][arm] = mean_metrics([r['metrics'] for r in rows.values()])
        for arm in ('P', 'TW4', 'AN4', 'P-keep', 'TW4-keep', 'AN4-keep'):
            changed = [ep for ep in EPISODES if not results[str(seed)][arm][str(ep)]['unchanged_from_T1']]
            gates[str(seed)][arm] = assess(arm, results[str(seed)], results[str(seed)]['T1'], changed)
    descriptive = {str(ep): bias_reading(data[ep]['start'], refs[ep][0, 0], data[ep]['stage_a'],
                                          data[ep]['registrations'], data[ep]['normal']) for ep in (26, 31)}
    stability = [{k: row.get(k) for k in ('episode', 'slot', 'name', 'admitted', 'stability_deg',
                                         'best_index', 'rival_index', 'rival_relative_margin', 'rival_separation_deg')}
                 for ep in EPISODES for row in data[ep]['registrations']]
    # Recompute T1, then compare seed-zero metrics with W1.0's immutable saved scores.
    heading = read_json(FEATURE / 'heading_queue.json')
    reproduction = {}
    for ep in EPISODES:
        parent = producer(heading['t1_scores'][str(ep)], 'pose_insurance.stereo_placement_diagnostic',
                          allowed=('public_label_diagnostic',))
        record['inputs'].append(input_file(ROOT, parent[0]))
        old = read_json(artifact(parent, 'metrics.json', record))['metrics']
        reproduction[str(ep)] = {k: abs(results['0']['T1'][str(ep)]['metrics'][k] - old[k]) for k in METRICS}
    residual = max(v for row in reproduction.values() for v in row.values())
    if residual > 1e-9:
        raise ValueError('T1 does not reproduce the original W1.0 scoring baseline')
    return dict(episodes=list(EPISODES), surface_seeds=[0, 1, 2], gated_surface_seed=0,
                disclosure='Disclosed train-label panel; not independent validation. Kinematic only, no policy or simulator. Scan samples approximate host vertices.',
                flip_stress_rule='(1/8) sum_episode mean_over_distinct_applied_combinations(metric); stipulated stress, not a probability; all recorded rivals retained.',
                frame0_seed_note='Frame-zero errors depend only on poses and labels, so are identical across surface seeds.',
                results=results, means=means, gates=gates, stability=stability, descriptive_bias=descriptive,
                P_equals_P_free_on_26_31=True,
                ranking=sorted(ARMS, key=lambda arm: means['0'][arm]['AUC'], reverse=True),
                reproduction=dict(max_residual=residual, tolerance=1e-9, episodes=reproduction), ambiguities=[])


def table(headers: list, rows: list) -> str:
    """Render a single tidy Markdown table."""
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def render(summary: dict) -> str:
    """Present each metric, pair limit, stability and nested gate clause in tables."""
    from ._panel_gates import ARMS, EPISODES, METRICS
    def fmt(value):
        return ('missing' if value is None or isinstance(value, float) and not math.isfinite(value)
                else f'{value:.6f}' if isinstance(value, float) else str(value))
    blocks = ['# Frame-zero panel Stage C', summary['disclosure'],
              'Gate decisions use seed 0 and original T1 over eight episodes. Keep arms are descriptive with no adoption route. P and P-free coincide on 26/31 and keep T1 on 0.',
              summary['flip_stress_rule'], summary['frame0_seed_note']]
    for seed in ('0', '1', '2'):
        blocks += [f'## Eight-episode means: surface seed {seed}',
            table(['Arm', *METRICS], [[arm, *[fmt(summary['means'][seed][arm][m]) for m in METRICS]] for arm in ARMS])]
        blocks += [f'## Every episode: surface seed {seed}',
            table(['Arm', 'Episode', *METRICS], [[arm, ep, *[fmt(summary['results'][seed][arm][str(ep)]['metrics'][m])
                    for m in METRICS]] for arm in ARMS for ep in EPISODES])]
    blocks += ['## Per-pair frame-zero errors and limits (all surface seeds)',
        table(['Arm', 'Episode', 'b residual cm', 'b angle deg', 'C deg', 'TW 5cm', 'AN 5cm', 'TW 3cm', 'AN 3cm'],
            [[arm, ep, fmt(row['frame0']['offset_error_m']*100), fmt(row['frame0']['b_error_deg']),
              fmt(row['frame0']['C_error_deg']), *[row['limits'][kind][key]
                for key in ('passed', 'passed_3cm') for kind in ('TW', 'AN')]]
             for arm in ARMS for ep in EPISODES for row in [summary['results']['0'][arm][str(ep)]]])]
    blocks += ['## HR2 subsampling stability',
        table(['Episode', 'Slot', 'Object', 'Admitted', 'Stability deg (three samples)', 'Rival index', 'Rival margin'],
            [[row['episode'], row['slot'], row['name'], row['admitted'],
              ', '.join(fmt(v) for v in row['stability_deg']) if row['stability_deg'] else 'missing',
              row['rival_index'], fmt(row['rival_relative_margin'])] for row in summary['stability']])]
    def clauses(node, prefix=''):
        rows = []
        if isinstance(node, dict):
            if 'passed' in node:
                rows.append([prefix, node['passed'], fmt(node.get('value')),
                             node.get('comparison', ''), fmt(node.get('threshold')), node.get('units', '')])
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    rows.extend(clauses(value, f'{prefix}/{key}'))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                if isinstance(value, (dict, list)):
                    rows.extend(clauses(value, f'{prefix}/{index}'))
        return rows
    for arm, gate in summary['gates']['0'].items():
        blocks += [f'## Gate {arm}: {"PASS" if gate["passed"] else "FAIL"}' + (' (descriptive only)' if gate['descriptive'] else ''),
                   'Baseline: immutable original T1; denominator 8. Frame-zero requirement: 2/3 required; 0 failed input admission. Pair 23 is previously inspected evidence.' if gate['kind'] != 'P' else 'Baseline: immutable original T1; denominator 8.',
                   table(['Clause', 'Pass', 'Value', 'Comparison', 'Threshold', 'Units'], clauses(gate['checks']))]
        if '6_flip_stress' in gate['checks']:
            stress = gate['checks']['6_flip_stress']
            blocks += [table(['Flip stress', *METRICS],
                         [[label, *[fmt(values[m]) for m in METRICS]] for label, values in
                          [('Expected means', stress['expected_means']), ('Change against T1', stress['changes'])]]),
                       table(['Episode', 'Distinct applied combinations'],
                             list(stress['distinct_combinations_per_episode'].items()))]
    blocks += ['## Centroid displacement diagnostics',
        table(['Episode', 'T1 b error cm', 'd + T1 rotation error cm', 'd + registered rotation error cm',
               'Offset length error cm (position lower bound)', 'C error deg (rotation only)'],
            [[ep, fmt(row['T1']['offset_error_m']*100), fmt(row['d_with_T1_rotation']['offset_error_m']*100),
              fmt(row['d_with_registered_rotation']['offset_error_m']*100), fmt(row['position_only_lower_bound_cm']),
              fmt(row['d_with_registered_rotation']['C_error_deg'])] for ep, row in summary['descriptive_bias'].items()]),
        table(['Episode', 'Slot', 'Object', 'Centroid minus registered origin, camera XYZ cm', 'Norm cm'],
            [[ep, b['slot'], b['name'], ', '.join(fmt(v*100) for v in b['camera_m']) if b['camera_m'] is not None else 'missing',
              fmt(b['norm_cm'])] for ep, row in summary['descriptive_bias'].items()
             for b in row['stereo_centroid_minus_registered_body_origin']]),
        next(iter(summary['descriptive_bias'].values()))['bias_caveat'],
        next(iter(summary['descriptive_bias'].values()))['attribution'],
        'No arm, threshold or failed construction was revised. No scored episode, policy, simulator or adoption action was run.']
    return '\n\n'.join(blocks) + '\n'


def run() -> None:
    """Check provenance, freeze all poses, score the fixed queue and publish once."""
    started = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    sys.dont_write_bytecode = True
    OUTPUT.mkdir(parents=True, exist_ok=True)
    check = preflight()
    runtime = runtime_flag(FEATURE, 'heading_authorized', 'Stage C CPU kinematic scoring')
    from ._panel_gates import EPISODES
    from src.paths import interpreter
    if Path(sys.executable).resolve() != interpreter(ROOT, runtime['diagnostic_python']):
        raise RuntimeError('Use the approved diagnostic interpreter')
    for ep in EPISODES:
        authorize_episode(runtime, SimpleNamespace(source_split='public', episode_index=ep), 'Stage C')
    record = dict(schema_version=1, kind='pose_insurance.panel_gates', status='running',
                  started_utc=stamp(), cpu_only=True, provenance='public_label_diagnostic',
                  preflight=check, inputs=[input_file(ROOT, PROTOCOL)],
                  sources=[input_file(ROOT, path) for path in sorted(FEATURE.glob('*.py'))] +
                          [input_file(ROOT, path) for path in
                           ('scripts/track3/static_pose_baseline.py', 'src/runs.py', 'src/paths.py',
                            'src/provenance.py', 'data/configs/sources.lock.json', 'data/configs/splits/public-v1.json')])
    data, _ = read_inputs(record)
    candidates = freeze(data, record['sources'])
    record.update(candidates=input_file(ROOT, OUTPUT / 'candidates.json'),
                  candidates_verified_before_labels_utc=stamp())
    # Preserve the earlier refusal byte-for-byte rather than edit a historical run.
    run_path = OUTPUT / 'run.json'
    if run_path.exists():
        old = read_json(run_path)
        if old.get('status') == 'refused_preflight':
            destination = OUTPUT / 'preflight-refusal' / 'run.json'
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                raise FileExistsError('Refusal archive already exists')
            run_path.rename(destination)
            record['prior_preflight_refusal'] = input_file(ROOT, destination)
        else:
            raise FileExistsError('A historical run.json exists; it cannot be overwritten')
    scores, refs = score_queue(candidates, data, record)
    summary = summarize(candidates, scores, data, refs, record)
    # Rehash all consumed inputs and source bytes before recording completion.
    record['inputs'] = list({row['path']: row for row in record['inputs']}.values())
    for entry in [*record['inputs'], *record['sources'], record['candidates']]:
        if sha256_file(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Consumed source/input changed during Stage C')
    record['final_preflight'] = preflight()
    summary['candidates'] = record['candidates']
    write_json(OUTPUT / 'summary.json', json_ready(summary))
    (OUTPUT / 'summary.md').write_text(render(summary))
    record.update(status='completed', finished_utc=stamp(), wall_seconds=time.monotonic()-started,
                  opened_label_episodes=list(EPISODES), forbidden_episodes_opened=[],
                  validation_episodes_opened=[], scored_episodes_opened=[],
                  label_free_candidate_order_enforced=True,
                  artifacts={name: sha256_file(OUTPUT/name) for name in
                             ('candidates.json', 'candidates.sha256', 'summary.json', 'summary.md', 'ledger.jsonl')})
    write_json(run_path, record)
    print(json.dumps(dict(status='completed', trials=record['trials'],
                          gates={arm: gate['passed'] for arm, gate in summary['gates']['0'].items()},
                          wall_seconds=record['wall_seconds'])), flush=True)


if __name__ == '__main__':
    run()
