"""Run each recalibrated, saved FP placement once through the static RL recipe."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from src.provenance import write_json
from .combined_frames import ROOT
from .frame0_fp import event
from .frame0_fp_policy import run
BASE=ROOT/'data/outputs/pose_insurance/frame0-fp/recalibrated'


def main():
    (BASE/'pid').write_text(str(os.getpid())+'\n')
    queue=json.loads((BASE/'queue.json').read_text())
    results=[]
    for episode in queue['episodes']:
        completed=BASE/f'ep{episode}/resumed-result.json'
        if completed.exists():
            results.append(json.loads(completed.read_text()));continue
        if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-11-04T21:15:00+00:00'):
            event(episode,'KEEP v5: reserve time for packing');break
        placement=json.loads((ROOT/f'data/outputs/pose_insurance/frame0-fp/corrected/ep{episode}/corrected-placement.json').read_text())
        event(episode,'static RL training START',recipe=dict(seed=0,steps=512,worlds=1,replay_minimum=32))
        try:
            result=run(episode,placement)
            event(episode,'static RL training/evaluation END; '+('ADMISSION PASS' if result['passed'] else 'KEEP v5 physical guard'),physical=result['physical'])
        except Exception as error:
            result=dict(episode=episode,passed=False,error=f'{type(error).__name__}: {error}')
            event(episode,'KEEP v5: policy operation failed',error=result['error'])
        write_json(completed,result);results.append(result)
        write_json(BASE/'training-results.json',dict(episodes=results,adopted=[r['episode'] for r in results if r['passed']]))
    write_json(BASE/'training-completed.json',dict(utc=datetime.now(timezone.utc).isoformat(),episodes=results,adopted=[r['episode'] for r in results if r['passed']]))


if __name__=='__main__':main()
