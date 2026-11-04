"""Admit the v5 hybrid against v4 and compare packed CSV text without losing signed zeros."""
import csv
from pathlib import Path
from src.paths import project_path
from src.provenance import input_file, sha256_file
from src.runs import ROOT, artifact, checked_entry, has_entry, producer, read_json, same_entry
from .scored_stereo_config import BUDGET, EPISODES, UNTOUCHED, POSE_PROVENANCE


def verify(manifest: dict) -> None:
    """Require one learned TW policy, its equivalent task and nineteen exact v4 exports."""
    from .scored_scale_handler import inspect
    from .scored_stereo_handler import admit_tw, reserved_tw
    expected = set((*EPISODES, *UNTOUCHED))
    episodes = [int(Path(row['path']).stem.split('_')[1]) for row in manifest['exports']]
    if (manifest['version'] != 'v5' or manifest['budget'] != BUDGET or len(episodes) != 20
            or set(episodes) != expected or manifest['composition'] != dict(TW=[19], v4=sorted(expected-{19}))):
        raise ValueError('v5 must change episode 19 only and retain nineteen exact v4 exports')
    for entry in [*manifest['inputs'], *manifest['gpu_runs'], manifest['reproduction_script'],
                  manifest['v4_parent'], manifest['tw_plan'], manifest['build'], manifest['cpu_inspection'], manifest['checkpoint']]:
        checked_entry(entry, {'inputs': []})
    plan = read_json(manifest['tw_plan']['path'])
    admit_tw(plan)
    selected = next(row for row in manifest['exports'] if Path(row['path']).stem == 'episode_000019')
    reservation = reserved_tw(plan, selected['run'])
    claim_entry = input_file(ROOT, 'data/outputs/pose_insurance/heading-v5/attempt-claim.json')
    claim = read_json(claim_entry['path'])
    if (claim['identity'] != plan['identity'] or claim['run'] != selected['run']
            or not same_entry(claim['reservation'], reservation)):
        raise ValueError('Selected TW policy differs from the exclusive attempt claim')
    parent = read_json(manifest['v4_parent']['path'])
    checked_entry(dict(path=parent['file'], sha256=parent['sha256']), {'inputs': []})
    parents = {int(Path(row['path']).stem.split('_')[1]): row for row in parent['exports']}
    for ep, row in zip(episodes, manifest['exports']):
        source = project_path(ROOT, row['source'])
        if sha256_file(source) != row['sha256'] or sha256_file(project_path(ROOT, row['path'])) != row['sha256']:
            raise ValueError(f'v5 trajectory hash mismatch: {ep}')
        if ep != 19 and (row['source'] != parents[ep]['path'] or row['sha256'] != parents[ep]['sha256'] or row['run'] != parents[ep]['run']):
            raise ValueError('A non-19 export differs from v4')
        policy = producer(row['run'], 'pose_insurance.run', allowed=POSE_PROVENANCE)
        if policy[1]['episode_index'] != ep or policy[1]['split'] != 'evaluation' or policy[1]['request']['budget'] != BUDGET:
            raise ValueError('Export policy differs from the episode or fixed budget')
        trajectory = artifact(policy, 'trajectory.parquet', {'inputs': []})
        build_entry = next(e for e in policy[1]['inputs'] if '/build' in e['path'] and e['path'].endswith('/run.json'))
        checked_entry(build_entry, {'inputs': []})
        task = producer(build_entry['path'], 'pose_insurance.build', allowed=POSE_PROVENANCE)
        if sha256_file(trajectory) != row['sha256'] or policy[1]['scene'] != task[1]['scene'] or not inspect(trajectory, task[1])['accepted']:
            raise ValueError('Policy export fails its admitted scene or physical checks')
        if ep == 19:
            if (source != trajectory or not source.resolve().is_relative_to(ROOT / 'data/outputs/pose_insurance/heading-v5/runs')
                    or build_entry != manifest['build'] or task[1]['scene'] != plan['scene']
                    or task[1]['equivalence'] != manifest['equivalence'] or not all(e['passed'] for e in manifest['equivalence'])
                    or policy[1]['plan_identity'] != plan['identity'] or policy[1]['request'].get('candidate') != 'TW'
                    or not has_entry(policy[1]['inputs'], reservation) or not has_entry(policy[1]['inputs'], claim_entry)
                    or claim['owner_pid'] != policy[1]['owner_pid']
                    or policy[1]['optimizer_updates'] < 1 or policy[1]['actor_adam_steps'] < 1
                    or dict(path=policy[1]['checkpoint'], sha256=policy[1]['checkpoint_sha256']) != manifest['checkpoint']):
                raise ValueError('TW requires the frozen equivalent build and a learned single policy')


def compare_csv(manifest: dict, path: Path) -> dict:
    """Compare all non-19 CSV fields as text, excluding only the globally changed commit URL."""
    parent = read_json(manifest['v4_parent']['path'])
    with project_path(ROOT, parent['file']).open(newline='') as stream:
        original_reader = csv.DictReader(stream)
        original = list(original_reader)
        fields = original_reader.fieldnames
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        current = list(reader)
        if reader.fieldnames != fields:
            raise ValueError('v5 CSV header differs from v4')
    if len(original) != 9120 or len(current) != 9120 or [r['row_id'] for r in original] != [r['row_id'] for r in current]:
        raise ValueError('v5 must preserve all 9,120 ordered CSV row IDs')
    compared = 0
    for before, after in zip(original, current):
        if before['row_id'].startswith('track_3/e000019/'):
            continue
        differences = [key for key in fields if key != 'code_commit_url' and before[key] != after[key]]
        if differences:
            raise ValueError(f"Non-19 CSV text changed at {before['row_id']}: {differences}")
        compared += 1
    if compared != 8650:
        raise ValueError('Expected exactly 8,650 non-19 CSV rows')
    return dict(passed=True, total_rows=9120, non_19_rows=compared, episode_19_rows=470,
                ignored_fields=['code_commit_url'], signed_zero_text_preserved=True,
                v4_csv=dict(path=parent['file'], sha256=parent['sha256']))
