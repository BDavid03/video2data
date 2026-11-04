"""Prepare only already-masked distinct instances on CPU while a policy trains."""

import json
import time

from src.provenance import write_json
from src.runs import now_utc
from . import two_object_campaign as campaign
from .two_object_instances import existing_pair
from .two_object_task import ROSTER


def main():
    queue=json.loads(campaign.s.QUEUE.read_text())
    queue['stop_utc']='2026-11-04T22:00:00+00:00'
    roster=json.loads(ROSTER.read_text())['episodes']
    completed=set()
    while True:
        control=campaign.read_control()
        if control['pause']:
            time.sleep(15)
            continue
        eligible=[]
        for episode in control['queue']:
            target=campaign.CAMPAIGN/f'ep{episode}/result.json'
            if episode in completed or target.exists() or episode in (1,):
                continue
            pair=existing_pair(episode) if episode not in (8,10) else None
            if episode in (8,10,19,25,15,14,20) or pair and pair['passed']:
                eligible.append(episode)
        if not eligible:
            break
        episode=eligible[0]
        campaign.s.event(2,f'ep{episode}: CPU-only preparation during training; saved tracks or distinct masks')
        row=campaign.prepare(episode,roster[str(episode)],queue)
        row['completed_utc']=now_utc()
        row['prepared_while_training']=True
        write_json(campaign.CAMPAIGN/f'ep{episode}/result.json',row)
        completed.add(episode)
        campaign.s.event(2,f'ep{episode}: CPU preparation {"PASS" if row["passed"] else "STOP"}: {row["reason"]}')


if __name__=='__main__':
    main()
