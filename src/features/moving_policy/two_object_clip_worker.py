"""Search separated clip instances and seed SAM at their first distinct frame."""

import json
from pathlib import Path
import sys

from src.features.moving_reference import _worker as original

BASE_SEGMENT=original.segment


def search(directory,job,torch):
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
    from ._two_object_instances import choose_pair
    root=original.ROOT
    weights=root/'data/.runtime/grounding-dino/weights'
    processor=AutoProcessor.from_pretrained(weights,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(weights,local_files_only=True).to('cuda').eval()
    rows=[]
    selected=None
    for index in range(5,job['frame_count'],5):
        original.check_time(job['rule'])
        image=Image.open(directory/'input_frames'/f'{index:06d}.jpg').convert('RGB')
        detections=[]
        for name in job['joint_names']:
            prompt=name.replace('_',' ')
            inputs=processor(images=image,text=prompt+'.',return_tensors='pt').to('cuda')
            with torch.inference_mode():
                raw=model(**inputs)
            result=processor.post_process_grounded_object_detection(raw,inputs.input_ids,
                box_threshold=job['rule']['box_threshold'],text_threshold=job['rule']['text_threshold'],
                target_sizes=[image.size[::-1]])[0]
            detections.append(dict(pan_attempts=[dict(frame_index=index,prompt=prompt,
                boxes=result['boxes'].detach().cpu().tolist(),scores=result['scores'].detach().cpu().tolist())]))
        chosen=choose_pair(detections)
        rows.append(dict(frame_index=index,detections=detections,selection=chosen))
        original.save_json(directory/'clip-search.partial.json',dict(rows=rows))
        if chosen['passed']:
            selected=dict(chosen,seed_frame=index)
            break
    result=dict(selected or dict(passed=False,reason='no separated joint box pair on five-frame cadence',box_iou=None),
                searched_frames=[r['frame_index'] for r in rows],rows=rows)
    original.save_json(directory/'clip-search.json',result)
    return dict(passed=result['passed'],seed_frame=result.get('seed_frame'),searched_frames=result['searched_frames'])


def segment(directory,job,torch):
    """Keep the original hand propagation and propagate the object from its own seed."""
    import numpy as np
    from sam2.build_sam import build_sam2_video_predictor
    found=json.loads((directory/'detections.json').read_text())
    original.save_json(directory/'detections-with-object.json',found)
    hand_found=dict(found,pan=None)
    original.save_json(directory/'detections.json',hand_found)
    try:
        counts=BASE_SEGMENT(directory,job,torch)
    finally:
        original.save_json(directory/'detections.json',found)
    rule=job['rule'];seed=found['pan']['frame_index']
    predictor=build_sam2_video_predictor(rule['sam_config'],str(original.ROOT/'data/.runtime/sam2/weights/sam2.1_hiera_large.pt'),
                                       device='cuda',mode='eval',apply_postprocessing=False)
    predictor.fill_hole_area=0
    object_counts=dict(forward=0,reverse=0)
    (directory/'frames').mkdir(exist_ok=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=predictor.init_state(str(directory/'input_frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
        try:
            predictor.add_new_points_or_box(state,frame_idx=seed,obj_id=0,box=np.asarray(found['pan']['box'],np.float32))
            for reverse in (False,True):
                for index,ids,logits in predictor.propagate_in_video(state,start_frame_idx=seed,reverse=reverse):
                    original.check_time(rule)
                    if reverse and index==seed:
                        continue
                    path=directory/'frames'/f'frame_{index:06d}.npz'
                    if path.exists():
                        with np.load(path) as saved:
                            arrays={k:saved[k].copy() for k in saved.files}
                    else:
                        arrays=dict(masks=np.zeros((3,*job['mask_shape']),bool),frame_index=np.int64(index),
                            timestamp_seconds=np.float64(job['timestamps'][index]),pts_integer=np.int64(job['pts_integer'][index]),
                            track_ids=np.arange(3,dtype=np.int32),selected_direction=np.zeros(3,np.uint8),
                            forward_reverse_iou=np.full(3,-1,np.float32),reverse_processed=np.bool_(False))
                    arrays['masks'][0]=(logits[0,0]>rule['logit_threshold']).cpu().numpy()
                    arrays['selected_direction'][0]=int(reverse)
                    arrays['reverse_processed']=np.bool_(bool(arrays['reverse_processed']) or reverse)
                    with path.with_suffix('.tmp').open('wb') as stream:
                        np.savez_compressed(stream,**arrays)
                    path.with_suffix('.tmp').replace(path)
                    object_counts['reverse' if reverse else 'forward']+=1
        finally:
            predictor.reset_state(state)
    return dict(counts,status='completed',object_seed_frame=seed,object_frames=object_counts)


def main():
    mode=sys.argv[1]
    original.detect=search if mode=='search' else original.detect
    if mode=='search':
        sys.argv[1]='detect'
    else:
        original.segment=segment
    original.main()


if __name__=='__main__':
    main()
