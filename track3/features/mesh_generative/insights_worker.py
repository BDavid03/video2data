"""Isolated Hunyuan generation worker owned by the mesh slice's GPU lease."""

import argparse
import json
import os
from pathlib import Path
import time

from track3.paths import PROJECT as ROOT, project_path
from track3.provenance import write_json
from track3.runs import runtime_flag

FEATURE = Path(__file__).parent


def generation_worker(request: str) -> None:
    """Repeat the frozen worker's pipeline, changing only its explicitly queued seed."""
    runtime_flag(FEATURE, 'insights_authorized', 'Mesh insights generation')
    lock = json.loads((ROOT/'.runtime/gpu-stage.lock').read_text())
    if lock['pid'] != os.getppid() or lock['stage'] != 'mesh_generative':
        raise PermissionError('Parent handler must own the GPU lease')
    job = json.loads(project_path(ROOT, request).read_text())
    import importlib.metadata
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if torch.__version__ != '2.7.1+cu128' or torch.version.cuda != '12.8':
        raise RuntimeError('Expected the frozen torch/CUDA versions')
    target = ROOT/job['mesh']
    if target.exists():
        raise FileExistsError('Refusing to overwrite generation')
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(ROOT/'.runtime/hunyuan-mesh/weights-xet'), subfolder='hunyuan3d-dit-v2-mini',
        variant='fp16', use_safetensors=True, dtype=torch.float16, device='cuda')
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    mesh = pipe(image=Image.open(ROOT/job['crop']).convert('RGB'), num_inference_steps=30,
                octree_resolution=380, mc_algo='mc',
                generator=torch.Generator(device='cuda').manual_seed(job['seed']))[0]
    torch.cuda.synchronize()
    finished = time.perf_counter()
    mesh.export(target)
    write_json(target.with_suffix('.json'), dict(seed=job['seed'], steps=30, octree_resolution=380,
               mc_algo='mc', load_seconds=loaded-start, generation_seconds=finished-loaded,
               total_seconds=finished-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
               peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
               vertices=len(mesh.vertices), faces=len(mesh.faces), device=torch.cuda.get_device_name(),
               preprocessing='white RGB crop; upstream ImageProcessorV2 defaults (border_ratio=0.15)',
               packages={p: importlib.metadata.version(p) for p in
                         ('torch', 'torchvision', 'diffusers', 'transformers', 'hy3dgen', 'scikit-image')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    generation_worker(parser.parse_args().request)
