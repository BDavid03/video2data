"""Isolated Grounding DINO worker for the one frozen frame-zero prompt."""

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
    lock = json.loads((root/'.runtime/gpu-stage.lock').read_text())
    if lock['pid'] != os.getppid() or lock['stage'] != 'mesh_generative':
        raise PermissionError('Mask inference requires the parent handler GPU lease')
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    job = json.loads(Path(args.request).read_text())
    rule = job['rule']
    if torch.__version__ != '2.7.1+cu128' or torch.version.cuda != '12.8':
        raise RuntimeError('DINO requires the pinned torch CUDA build')
    output = root/job['output']
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    weights = root/'.runtime/grounding-dino/weights'
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(weights, local_files_only=True).to('cuda').eval()
    image = Image.open(root/job['image']).convert('RGB')
    inputs = processor(images=image, text=rule['prompt'], return_tensors='pt').to('cuda')
    with torch.inference_mode():
        prediction = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        prediction, inputs.input_ids, box_threshold=rule['box_threshold'],
        text_threshold=rule['text_threshold'], target_sizes=[image.size[::-1]])[0]
    scores = results['scores'].detach().cpu()
    boxes = results['boxes'].detach().cpu()
    index = int(scores.argmax()) if len(scores) else None
    torch.cuda.synchronize()
    report = dict(status='completed', box=boxes[index].tolist() if index is not None else None,
                  box_score=float(scores[index]) if index is not None else None, selected_index=index,
                  boxes=boxes.tolist(), scores=scores.tolist(), rule=rule,
                  wall_seconds=time.perf_counter()-start,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  packages={p:importlib.metadata.version(p) for p in ('torch','torchvision','transformers','huggingface-hub','tokenizers')})
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
