"""Screen the amended joint queue and schedule every gate-passing policy once."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2

from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import two_object_screen as s
from . import two_object_reference as reference
from .two_object_task import ROSTER

ROOT = s.ROOT
OUTPUT = ROOT/'data/outputs/moving_policy/two-object'
CAMPAIGN = OUTPUT/'campaign-facing-table'
ORDER = [19,25,15,30,32,33,3,5,6,14,20]
CUTOFF = datetime.fromisoformat('2026-11-04T19:45:00+00:00')


def final_waivers():
    decision = OUTPUT/'fraction-waiver-decision.json'
    if not decision.exists():
        return None
    value = json.loads(decision.read_text())
    return value['final_M1_waived']


def render(rows, state):
    lines=['# Two-object campaign: disclosed deadline, budget and texture-fraction amendments','',state,'',
           'Instance method: joint minimum box IoU <0.5; frame-0 mask IoU <0.5. Existing distinct masks may be reused.', '',
           '| Episode | A | B | Stage | First failed gate / result |',
           '| ---: | --- | --- | --- | --- |']
    latest={}
    for base in ('campaign','campaign-30g','campaign-camera-regions','campaign-facing-table'):
        for path in sorted((OUTPUT/base).glob('ep*/result.json')):
            r=json.loads(path.read_text());latest[r['episode']]=r
    latest.update({r['episode']:r for r in rows})
    rows=list(latest.values())
    for r in rows:
        lines.append(f'| {r["episode"]} | {r.get("A","unselected")} | {r.get("B","unselected")} | {r["stage"]} | {r["reason"]} |')
    lines += ['','Both mask-usability measures are waived; stereo support >=0.9 and all other reference/physical limits remain.',
              'Last training start 19:45 UTC; pack target 20:50 UTC; close 22:00 UTC. No scored labels or scans.']
    (OUTPUT/'summary.md').write_text('\n'.join(lines)+'\n')
    write_json(OUTPUT/'summary.json',dict(status=state,episodes=rows,updated_utc=now_utc(),queue=read_control()['queue']))


def prepare(episode, names, queue):
    directory=CAMPAIGN/f'ep{episode}'
    directory.mkdir(parents=True,exist_ok=True)
    result=directory/'result.json'
    if result.exists():
        return json.loads(result.read_text())
    attempts=directory/'attempts'
    attempts.mkdir(exist_ok=True)
    with (attempts/f'{time.time_ns()}.json').open('x') as stream:
        json.dump(dict(episode=episode,started_utc=now_utc(),amendment=sha256_file(OUTPUT/'campaign-amendment.json')),
                  stream,indent=2)
    if episode in (8,10):
        from .two_object_rescore import run as rescore
        return rescore(episode,directory)
    row=dict(episode=episode,passed=False,stage='detection',A='unselected',B='unselected')
    from .two_object_clip_instances import run as instances
    prior=None
    if episode in (19,25,15,14,20):
        previous=next(OUTPUT/base/f'ep{episode}' for base in ('campaign-camera-regions','campaign-30g','campaign')
                      if (OUTPUT/base/f'ep{episode}/result.json').exists())
        prior=json.loads((previous/'result.json').read_text())
        separated=prior['instance_separation']
        write_json(directory/'reference-rerun-reuse.json',dict(previous_result=str((previous/'result.json').relative_to(ROOT)),
            previous_result_sha256=sha256_file(previous/'result.json'),instance_separation=separated,
            protocol_sha256=sha256_file(ROOT/'docs/chain/moving-two-object-protocol.md'),method='clip facing, own-mesh flip and available table patches; reference-only rerun; camera solve reused'))
    else:
        separated=instances(episode,names,queue)
    row['instance_separation']=separated
    if not separated['passed']:
        return dict(row,stage='instance-separation',reason=f'{separated["reason"]}; box IoU={separated.get("box_iou")}; frame-0 mask IoU={separated.get("frame0_mask_iou")}')
    mask_dirs=[ROOT/d for d in separated['directories']]
    old_role=previous/'roles.json' if prior else OUTPUT/f'ep{episode}/roles.json'
    current_role=directory/'roles.json'
    role=json.loads(current_role.read_text()) if current_role.exists() else json.loads(old_role.read_text()) if (prior or separated['reused']) and old_role.exists() else s.roles(mask_dirs)
    write_json(directory/'roles.json',role)
    if not role['passed']:
        return dict(row,stage='roles',reason=role['reason'])
    active=role['active_slot']
    row.update(A=names[active],B=names[1-active],active_slot=active)
    selected=s.selected_inputs(episode,mask_dirs[active])
    old_gate=previous/'mask-gates.json' if prior else OUTPUT/f'ep{episode}/mask-gates.json' if separated['reused'] else directory/'new-instance-mask-gates.json'
    saved_gate=old_gate if old_gate.exists() else directory/'mask-gates.json'
    measured=json.loads(saved_gate.read_text()) if saved_gate.exists() else s.mask_gates(selected)
    write_json(directory/'mask-gates.json',dict(measured,waived=['mask_usable_fraction','mask_longest_unusable_interval_frames'],
        actual_source=str(old_gate.relative_to(ROOT)) if old_gate.exists() else 'fresh campaign measurement'))
    for name,g in measured['gates'].items():
        if name not in ('mask_usable_fraction','mask_longest_unusable_interval_frames') and not g['passed']:
            return dict(row,stage='mask-gates',gates=measured['gates'],
                reason=f'{name}: {g["value"]:.8g} {g["comparison"]} {g["threshold"]}')
    waivers=final_waivers()
    if waivers is None and not measured['gates']['mask_usable_fraction']['passed']:
        return dict(row,stage='pending-clarification',reason='Final M1 duplicate texture-fraction waiver needs owner clarification')
    final=reference.run(episode,selected,directory/'candidate-c',mask_dirs[1-active],
                        camera_source=previous/'candidate-c' if prior else None)
    if waivers:
        final['waived_limits']=waivers
        final['passed']=all(g['passed'] for name,g in final['gates'].items() if name not in waivers)
        write_json(directory/'candidate-c/campaign-gates-final.json',final)
    failed=[(n,g) for n,g in final['gates'].items() if n not in (waivers or []) and not g['passed'] and g.get('status')!='not_run_after_first_failure']
    reason='; '.join(f'{n}: {g["value"]} {g["comparison"]} {g["threshold"]}' for n,g in failed[:1])
    row.update(stage='reference',passed=final['passed'],gates=final['gates'],reason='PASS' if final['passed'] else reason or final['reason'])
    if final['passed']:
        from .two_object_stage0 import prepare as stage0
        physical=stage0(episode,active,directory/'candidate-c/reference.npz',directory/'policy')
        row.update(stage='Stage 0',passed=physical['status']=='PASS',reason=physical['reason'])
    return row


def read_control():
    control=json.loads((OUTPUT/'control.json').read_text())
    allowed={1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33}
    order=control['queue']
    if (type(control['pause']) is not bool or not isinstance(control['note'],str)
            or not isinstance(order,list) or any(type(e) is not int or e not in allowed for e in order)
            or len(order)!=len(set(order))):
        raise ValueError('Invalid control: require pause bool, allowed unique integer episode queue, note string')
    return control


def main():
    cv2.setNumThreads(2)
    cv2.setRNGSeed(31)
    CAMPAIGN.mkdir(exist_ok=True)
    request=dict(order=ORDER,started_utc=now_utc(),source_sha256=sha256_file(Path(__file__)),
                 amendment_sha256=sha256_file(OUTPUT/'campaign-amendment.json'))
    with (CAMPAIGN/f'execution-{time.time_ns()}.json').open('x') as stream:
        json.dump(request,stream,indent=2)
    queue=json.loads(s.QUEUE.read_text())
    queue.update(stop_utc='2026-11-04T22:00:00+00:00')
    roster=json.loads(ROSTER.read_text())['episodes']
    rows=[]
    render(rows,'Screen running; no training yet.')
    try:
        completed=set()
        while True:
            control=read_control()
            if control['pause']:
                s.event('PAUSE',f'campaign paused between episodes: {control["note"]}')
                time.sleep(15)
                continue
            pending=[e for e in control['queue'] if e not in completed]
            if not pending:
                break
            episode=pending[0]
            s.event('CONTROL',f'next ep{episode}; note={control["note"]}; control SHA256={sha256_file(OUTPUT/"control.json")}')
            s.event(2,f'campaign ep{episode}: mask/reference/Stage-0 screen started')
            row=prepare(episode,roster[str(episode)],queue)
            row['completed_utc']=now_utc()
            write_json(CAMPAIGN/f'ep{episode}/result.json',row)
            with (CAMPAIGN/'ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            rows.append(row)
            completed.add(episode)
            render(rows,'Screen running; see training-queue.json for policy state.')
            s.event(2,f'campaign ep{episode}: {"PASS" if row["passed"] else "STOP"} at {row["stage"]}: {row["reason"]}')
            if row['passed']:
                q=json.loads((OUTPUT/'training-queue.json').read_text())
                q['ready']=[r for r in q['ready'] if r['episode']!=episode]
                q['ready'].append(dict(episode=episode,policy=str((CAMPAIGN/f'ep{episode}/policy').relative_to(ROOT)),ready_utc=now_utc()))
                q['updated_utc']=now_utc()
                write_json(OUTPUT/'training-queue.json',q)
                if datetime.now(timezone.utc)<CUTOFF:
                    from .two_object_run import run as policy
                    policy(episode,CAMPAIGN/f'ep{episode}/policy')
                else:
                    s.event(3,f'ep{episode}: ready after 19:45 UTC; no training launch')
        render(rows,'Screen complete. Training and pack outcomes are in training-queue.json.')
        write_json(CAMPAIGN/'completed.json',dict(time_utc=now_utc(),episodes=len(rows),ready=sum(r['passed'] for r in rows)))
    except BaseException as error:
        render(rows,f'Campaign stopped: {type(error).__name__}: {error}')
        write_json(CAMPAIGN/'error.json',dict(time_utc=now_utc(),episode=episode,error=f'{type(error).__name__}: {error}'))
        s.event('STOP',f'campaign ep{episode}: {type(error).__name__}: {error}; no automatic retry')
        raise


if __name__=='__main__':
    main()
