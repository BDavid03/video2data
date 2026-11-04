"""Pure joint box assignment and object-mask overlap measurements."""

import numpy as np
from src.features.moving_reference._masks import box_iou


def choose_pair(detections, threshold=.4):
    candidates=[]
    for detection in detections:
        rows=[]
        for attempt in detection['pan_attempts']:
            for index,(box,score) in enumerate(zip(attempt['boxes'],attempt['scores'])):
                if score>threshold:
                    b=np.asarray(box,float)
                    rows.append(dict(box=box,score=score,prompt=attempt['prompt'],selected_index=index,
                                     area=float(np.maximum(b[2:]-b[:2],0).prod())))
        candidates.append(rows)
    pairs=[]
    for i,a in enumerate(candidates[0]):
        for j,b in enumerate(candidates[1]):
            overlap=box_iou(np.asarray(a['box'],float),np.asarray(b['box'],float))
            pairs.append(((overlap,a['area'],b['area'],i,j),a,b))
    if not pairs:
        return dict(passed=False,reason='no qualifying DINO pair',box_iou=None)
    key,a,b=min(pairs,key=lambda p:p[0])
    return dict(passed=key[0]<.5,box_iou=float(key[0]),selected=[a,b],
                reason='distinct joint boxes' if key[0]<.5 else 'no box pair with IoU <0.5')


def mask_overlap(a,b):
    a=np.asarray(a,bool);b=np.asarray(b,bool)
    union=int((a|b).sum())
    return float((a&b).sum()/union) if union else 1.
