"""Frame-zero object boxes (Grounding DINO) or masks (SAM2.1 image predictor) for one saved frame.

Usage: _frame0_masks_worker.py detect <job.json>   (grounding-dino venv)
       _frame0_masks_worker.py segment <job.json>  (sam2 venv)
The job holds: image (jpg), names, prompts, box_threshold, text_threshold, output directory, boxes (segment only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def detect(job):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    weights = ROOT / 'data/.runtime/grounding-dino/weights'
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(weights, local_files_only=True).to('cuda').eval()
    image = Image.open(job['image']).convert('RGB')
    attempts = []
    for name, prompt in zip(job['names'], job['prompts']):
        inputs = processor(images=image, text=prompt.rstrip('.') + '.', return_tensors='pt').to('cuda')
        with torch.inference_mode():
            raw = model(**inputs)
        found = processor.post_process_grounded_object_detection(
            raw, inputs.input_ids, box_threshold=job['box_threshold'], text_threshold=job['text_threshold'],
            target_sizes=[image.size[::-1]])[0]
        attempts.append(dict(name=name, prompt=prompt, boxes=found['boxes'].cpu().tolist(), scores=found['scores'].cpu().tolist()))
    Path(job['output']).joinpath('detections.json').write_text(json.dumps(dict(attempts=attempts), indent=2))


def segment(job):
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', str(ROOT / 'data/.runtime/sam2/weights/sam2.1_hiera_large.pt'), device='cuda')
    predictor = SAM2ImagePredictor(model)
    image = np.asarray(Image.open(job['image']).convert('RGB'))
    masks = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictor.set_image(image)
        for box in job['boxes']:
            m, scores, _ = predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=False)
            masks.append(m[0].astype(bool))
    np.savez_compressed(Path(job['output']) / 'masks.npz', masks=np.stack(masks), names=np.asarray(job['names']))


if __name__ == '__main__':
    operation, path = sys.argv[1:3]
    {'detect': detect, 'segment': segment}[operation](json.loads(Path(path).read_text()))
