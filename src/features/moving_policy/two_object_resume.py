"""Resume the frozen screen once after the lead's disclosed manifest repair."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import cv2

from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import two_object_screen as screen
from . import two_object_reference as reference
from .two_object_task import PROTOCOL, ROSTER

ROOT = screen.ROOT
OUTPUT = ROOT/'data/outputs/moving_policy/two-object'
TRAINING_CUTOFF = datetime.fromisoformat('2026-11-04T15:35:00+00:00')


def verify_repair():
    """Bind the restored manifests and only the previously disclosed guard change."""
    moving = json.loads((ROOT/'data/outputs/pose_insurance/combined-v5/release-moving-3/track3_combined_v5.manifest.json').read_text())
    entries = moving['parents']
    restored = []
    for name, digest in [
        ('submission/20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json',
         '0fcbfd7d48dc137fc5598adbcb8d609c5de3dd717a87ceb886b9b885e7933f7d'),
        ('submission/20261103T2058Z-d1-v5/track3_meshes_v5_final.manifest.json',
         'fe9d085fd6dd16e0c2cefd01bb5823cd4336a2e59ceaeeffb96fddcdeee93c69')]:
        if sha256_file(ROOT/name) != digest or not any(e['path']==name and e['sha256']==digest for e in entries):
            raise ValueError('Restored parent differs from uploaded moving manifest')
        restored.append(dict(path=name,sha256=digest))
    previous = json.loads((OUTPUT/'identity.json').read_text())
    correction = json.loads((OUTPUT/'reference-boundary-correction.json').read_text())
    for name, digest in previous['sources'].items():
        expected = correction['after_sha256'] if name=='two_object_reference.py' else digest
        if sha256_file(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Unapproved source change before resume: {name}')
    protocol_hash = subprocess.check_output(['git','hash-object',str(PROTOCOL)],cwd=ROOT,text=True).strip()
    if protocol_hash != previous['protocol_hash_object']:
        raise ValueError('Frozen scientific protocol changed')
    return dict(parents=restored,original_identity=previous,guard=correction,
                resume_handler_sha256=sha256_file(Path(__file__)),time_utc=now_utc(),
                decision='Lead-authorized one-time resume at episode 3; no episode-1 retry; continue screening after training cutoff')


def table(rows, status):
    """Report the first failed gate with its measurements for every finished episode."""
    lines = ['# Two-object screen: disclosed infrastructure resume','',
             status,'','| Episode | A | B | Stage | First failed gate / result |',
             '| ---: | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f'| {row["episode"]} | {row.get("A", "unselected")} | {row.get("B", "unselected")} | {row["stage"]} | {row["reason"]} |')
    lines += ['','Training cutoff: 2026-11-04 15:35 UTC. Screening continues as evidence after that cutoff.',
              'No scored labels, scans, threshold changes, re-prompts or episode retries. Episode 1 is recorded from its saved inputs only.']
    (OUTPUT/'summary.md').write_text('\n'.join(lines)+'\n')
    write_json(OUTPUT/'summary.json',dict(status=status,episodes=rows,completed_utc=now_utc(),
                                        training_cutoff_utc=TRAINING_CUTOFF.isoformat()))


def attempt(episode, names, queue):
    """Use the existing mask, role, gate and reference handlers without changing them."""
    directory = OUTPUT/f'ep{episode}'
    directory.mkdir(exist_ok=True)
    with (directory/'attempt.json').open('x') as stream:
        json.dump(dict(episode=episode,started_utc=now_utc(),resume='lead-authorized',
                       guard_sha256=sha256_file(Path(reference.__file__))),stream)
    row = dict(episode=episode,passed=False,stage='detection',A='unselected',B='unselected')
    mask_dirs = [directory/f'masks-{i}' for i in (0,1)]
    for name, target in zip(names,mask_dirs):
        record = screen.run_masks(episode,name,target,queue)
        if record['status'] != 'completed':
            return dict(row,reason=f'no qualifying detection: {name}')
    role = screen.roles(mask_dirs)
    write_json(directory/'roles.json',role)
    if not role['passed']:
        return dict(row,stage='roles',reason=role['reason'])
    active = role['active_slot']
    row.update(A=names[active],B=names[1-active],active_slot=active,
               motion_p95_image_width=role['displacement_p95_image_width'])
    screen.event(2,f'ep{episode}: A={row["A"]}, B={row["B"]}; p95={row["motion_p95_image_width"]}')
    selected = screen.selected_inputs(episode,mask_dirs[active])
    measured = screen.mask_gates(selected)
    write_json(directory/'mask-gates.json',measured)
    for name, gate in measured['gates'].items():
        if not gate['passed']:
            return dict(row,stage='mask-gates',reason=f'{name}: {gate["value"]:.8g} {gate["comparison"]} {gate["threshold"]}',
                        gates=measured['gates'])
    result = reference.run(episode,selected,directory/'candidate-c',mask_dirs[1-active])
    failed = [(name,g) for name,g in result['gates'].items() if not g['passed'] and g.get('value') is not None]
    detail = '; '.join(f'{name}: {g["value"]:.8g} {g["comparison"]} {g["threshold"]}' for name,g in failed[:1])
    return dict(row,stage='reference',passed=result['passed'],gates=result['gates'],
                reason='PASS' if result['passed'] else detail or result['reason'])


def main():
    cv2.setNumThreads(2)
    cv2.setRNGSeed(31)
    decision = verify_repair()
    with (OUTPUT/'resume-once.json').open('x') as stream:
        json.dump(decision,stream,indent=2)
    with screen.STATUS.open('a') as stream:
        stream.write(f'RESUME {now_utc()}: infrastructure fault repaired (manifests restored, hashes equal); ep1 STOP at reference (0 visible A pixels at frame 0); queue continues at ep3; no threshold, rule or code change except the fail-closed guard already recorded\n')
    queue = json.loads(screen.QUEUE.read_text())
    # The lead permits evidence-only screening after the submission clock closes.
    queue['stop_utc'] = '2026-11-05T02:00:00+00:00'
    write_json(OUTPUT/'resume-execution.json',dict(screen_worker_deadline_utc=queue['stop_utc'],
        training_cutoff_utc=TRAINING_CUTOFF.isoformat(),
        note='Execution deadlines only; frozen scientific queue, models, detector prompts and thresholds are unchanged'))
    rows = [dict(episode=1,A='white_pot',B='white_pot_lid',active_slot=0,stage='reference',passed=False,
                 reason='frame-0 visible A stereo pixels after B exclusion: 0; M1 initialization unavailable',
                 decision='lead-authorized terminal result from saved input audit; no rerun')]
    ep1 = OUTPUT/'ep1'
    if not (ep1/'result.json').exists():
        write_json(ep1/'result.json',rows[0])
    if not (ep1/'candidate-c/gates-final.json').exists():
        reference.stopped(1,ep1/'candidate-c',rows[0]['reason'],{})
    roster = json.loads(ROSTER.read_text())['episodes']
    table(rows,'Screen running from episode 3; no training or pack produced.')
    try:
        for episode in queue['episodes'][1:]:
            target = OUTPUT/f'ep{episode}/result.json'
            if target.exists():
                row = json.loads(target.read_text())
            else:
                row = attempt(episode,roster[str(episode)],queue)
                row['completed_utc'] = now_utc()
                write_json(target,row)
                with (OUTPUT/'ledger.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
            rows.append(row)
            table(rows,'Screen running; training has not run.')
            screen.event(2,f'ep{episode}: {"PASS" if row["passed"] else "STOP"} at {row["stage"]}: {row["reason"]}')
            if row['passed'] and datetime.now(timezone.utc) < TRAINING_CUTOFF:
                write_json(OUTPUT/'training-ready.json',row)
                table(rows,'First passing reference before training cutoff; joint Stage 0 required before training.')
                return
        table(rows,'Screen complete; no training or candidate pack produced by this screen.')
        write_json(OUTPUT/'resume-completed.json',dict(time_utc=now_utc(),episodes=len(rows),
                   passing_references=sum(r['passed'] for r in rows),training_run=False,candidate_pack=None))
        screen.event(2,f'full screen complete: {len(rows)} episodes; {sum(r["passed"] for r in rows)} reference passes')
    except BaseException as error:
        table(rows,f'Screen stopped at infrastructure error: {type(error).__name__}: {error}')
        write_json(OUTPUT/'resume-error.json',dict(time_utc=now_utc(),error=f'{type(error).__name__}: {error}'))
        screen.event('STOP',f'resume infrastructure error: {type(error).__name__}: {error}; no retry')
        raise


if __name__ == '__main__':
    main()
