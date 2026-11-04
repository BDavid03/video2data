"""Derive the frame-zero adoption limits from the train registrations and apply them once to saved scored registrations.

Run: PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_refresh.calibrate <refresh dir>
The rule is fixed in <refresh dir>/protocol.md (hash in protocol.sha256) before any result was seen.
"""
import json
import sys
from pathlib import Path

from src.provenance import sha256_file, write_json
from src.runs import now_utc
from .frame0_fp import ROOT

SCORED = (8, 10, 14, 19, 20, 25, 30)
SAVED = ROOT / 'data/outputs/pose_insurance/frame0-fp/corrected'


def threshold(values, valid, higher_is_better):
    """Best-separating threshold over observed values; ties toward the stricter value; all valid -> worst valid value."""
    pairs = [(v, ok) for v, ok in zip(values, valid) if v is not None]
    if all(ok for _, ok in pairs):
        return min(v for v, _ in pairs) if higher_is_better else max(v for v, _ in pairs)
    best = None
    for candidate, _ in pairs:
        passes = [(v >= candidate) if higher_is_better else (v <= candidate) for v, _ in pairs]
        accuracy = sum(p == ok for p, (_, ok) in zip(passes, pairs))
        strictness = candidate if higher_is_better else -candidate
        if best is None or (accuracy, strictness) > best[0]:
            best = ((accuracy, strictness), candidate)
    return best[1]


def pair_values(rows):
    """Per pair: min IoU over both objects, max depth residual over both, B's rival margin."""
    regs = rows['registrations']
    return (min(r['iou'] for r in regs), max(r['depth_m'] for r in regs if r['depth_m'] is not None) if any(r['depth_m'] is not None for r in regs) else None,
            regs[1]['margin'] if regs[1]['margin'] is not None else float('inf'))


def main():
    base = Path(sys.argv[1]).resolve()
    protocol = base / 'protocol.md'
    if sha256_file(protocol) != (base / 'protocol.sha256').read_text().split()[0]:
        raise ValueError('Protocol changed after sealing')
    train = [r for r in json.loads((base / 'train/results.json').read_text())['episodes'] if 'registrations' in r]
    valid = [r['validation']['valid'] for r in train]
    values = [pair_values(r) for r in train]
    limits = dict(mask_iou_min=threshold([v[0] for v in values], valid, True),
                  depth_residual_max_m=threshold([v[1] for v in values], valid, False),
                  rival_margin_min=threshold([v[2] for v in values], valid, True),
                  train_pairs=[dict(episode=r['episode'], valid=r['validation'], values=v) for r, v in zip(train, values)],
                  protocol_sha256=sha256_file(protocol), utc=now_utc())
    write_json(base / 'limits.json', limits)
    decisions = []
    for episode in SCORED:
        path = SAVED / f'ep{episode}/corrected-placement.json'
        placement = json.loads(path.read_text())
        regs = placement['registrations']
        failures = []
        for r in regs:
            if r['silhouette_iou'] < limits['mask_iou_min']:
                failures.append(f"{r['name']} IoU")
            if r['median_depth_residual_m'] is None or r['median_depth_residual_m'] > limits['depth_residual_max_m']:
                failures.append(f"{r['name']} depth")
        margin = regs[1]['distinct_rival_relative_margin']
        if margin is not None and margin < limits['rival_margin_min']:
            failures.append('B rival margin')
        decisions.append(dict(episode=episode, passed=not failures, failures=failures, placement=str(path.relative_to(ROOT)),
                              placement_sha256=sha256_file(path),
                              measurements=[dict(name=r['name'], iou=r['silhouette_iou'], depth_m=r['median_depth_residual_m'],
                                                 margin=r['distinct_rival_relative_margin']) for r in regs]))
    ready = [d['episode'] for d in sorted((d for d in decisions if d['passed']), key=lambda d: -(d['measurements'][1]['margin'] or 0))]
    write_json(base / 'decisions.json', dict(limits=limits, episodes=decisions, ready=ready, utc=now_utc()))
    print(json.dumps(dict(limits={k: limits[k] for k in ('mask_iou_min', 'depth_residual_max_m', 'rival_margin_min')}, ready=ready,
                          failures={d['episode']: d['failures'] for d in decisions}), indent=1))


if __name__ == '__main__':
    main()
