"""Isolated SAM2.1 image predictor with a single box and no mask postprocessing."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    lock = json.loads((root/'data/.runtime/gpu-stage.lock').read_text())
    if lock['pid'] != os.getppid() or lock['stage'] != 'mesh_generative':
        raise PermissionError('Mask inference requires the parent handler GPU lease')
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    job = json.loads(Path(args.request).read_text())
    if torch.__version__ != '2.7.1+cu128' or torch.version.cuda != '12.8':
        raise RuntimeError('SAM2 requires the pinned torch CUDA build')
    output = root/job['output']
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = build_sam2(job['rule']['config'], str(root/'data/.runtime/sam2/weights/sam2.1_hiera_large.pt'),
                       device='cuda', mode='eval', apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model, max_hole_area=0.0, max_sprinkle_area=0.0)
    with torch.inference_mode():
        predictor.set_image(np.asarray(Image.open(root/job['image']).convert('RGB')))
        masks, scores, _ = predictor.predict(box=np.array(job['box']), multimask_output=True)
    index = int(np.argmax(scores))
    np.savez_compressed(output, mask=masks[index].astype(bool))
    torch.cuda.synchronize()
    report = dict(status='completed', selected_index=index, predicted_ious=scores.tolist(),
                  selected_predicted_iou=float(scores[index]), rule=job['rule'],
                  wall_seconds=time.perf_counter()-start,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  packages={p:importlib.metadata.version(p) for p in ('torch','torchvision','SAM-2')})
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
