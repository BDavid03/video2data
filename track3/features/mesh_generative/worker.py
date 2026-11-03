"""Isolated shape-only Hunyuan worker; invoked by the gated lease-owning handler."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    lock = json.loads((root/'.runtime/gpu-stage.lock').read_text())
    if lock['pid'] != os.getppid() or lock['stage'] != 'mesh_generative':
        raise PermissionError('The worker requires its parent handler to own the GPU lease')
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if torch.__version__ != '2.7.1+cu128' or torch.version.cuda != '12.8':
        raise RuntimeError(f'Expected torch 2.7.1+cu128 / CUDA 12.8; got {torch.__version__} / {torch.version.cuda}')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('One mesh only; refusing to overwrite an existing generation')
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.weights, subfolder='hunyuan3d-dit-v2-mini', variant='fp16',
        use_safetensors=True, dtype=torch.float16, device='cuda')
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    mesh = pipe(image=Image.open(args.image).convert('RGB'),
                num_inference_steps=30, octree_resolution=380, mc_algo='mc',
                generator=torch.Generator(device='cuda').manual_seed(12345))[0]
    torch.cuda.synchronize()
    finished = time.perf_counter()
    mesh.export(output)
    report = dict(load_seconds=loaded-start, generation_seconds=finished-loaded,
                  total_seconds=finished-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  vertices=len(mesh.vertices), faces=len(mesh.faces), device=torch.cuda.get_device_name(),
                  seed=12345, steps=30, octree_resolution=380, mc_algo='mc',
                  preprocessing='white RGB crop; upstream ImageProcessorV2 defaults (border_ratio=0.15)',
                  packages={p: importlib.metadata.version(p) for p in
                            ('torch','torchvision','diffusers','transformers','hy3dgen','scikit-image')})
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
