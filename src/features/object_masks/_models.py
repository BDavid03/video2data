"""Optional offline detector and video segmenter for the mask feature."""

from contextlib import contextmanager
import importlib.metadata
import os
from pathlib import Path
import re
import sys

from src.paths import project_path
from src.provenance import input_file
from src.runs import ROOT, read_json
from .opening_clusters import FEATURE

CACHE_DIRECTORIES = {"TORCH_HOME": "data/.cache/object-masks/torch", "HF_HOME": "data/.cache/object-masks/huggingface",
                     "XDG_CACHE_HOME": "data/.cache/object-masks", "MPLCONFIGDIR": "data/.cache/object-masks/matplotlib",
                     "TMPDIR": "data/.runtime/tmp/object-masks", "TMP": "data/.runtime/tmp/object-masks",
                     "TEMP": "data/.runtime/tmp/object-masks"}


def _pinned(items, record, inside=None):
    """Hash-check an installation inventory; returns {resolved path: input entry}."""
    pinned = {}
    for item in items:
        path = project_path(ROOT, item["path"])
        if path in pinned or (inside is not None and not path.is_relative_to(inside)):
            raise ValueError("Installed mask files must be unique and inside the environment")
        entry = input_file(ROOT, path)
        if entry["sha256"] != item["sha256"]:
            raise ValueError(f"Installed mask file differs from its recorded hash: {path}")
        pinned[path] = entry
        record["inputs"].append(entry)
    return pinned


def provisioned(record):
    config = read_json(FEATURE / "backends.json")
    environment = project_path(ROOT, config["environment"])
    if Path(sys.prefix).resolve() != environment:
        raise PermissionError("Use the isolated project object-mask environment")
    if not config.get("installation_record") or not config.get("installation_record_sha256"):
        raise PermissionError("Mask model dependencies and assets have not been approved and provisioned")
    entry = input_file(ROOT, project_path(ROOT, config["installation_record"]))
    if entry["sha256"] != config["installation_record_sha256"]:
        raise ValueError("Mask installation record differs from its pin")
    record["inputs"].append(entry)
    installed = read_json(entry["path"])
    if (installed.get("status") != "completed" or installed.get("user_approved") is not True or
            installed.get("backend") != "grounding_dino_sam2" or
            project_path(ROOT, installed["environment"]) != environment):
        raise PermissionError("A completed approved mask installation is required")
    for key in ("detector_revision", "sam2_source_commit", "sam2_model_revision"):
        if not isinstance(config.get(key), str) or not re.fullmatch(r"[a-f0-9]{40}", config[key]) or installed.get(key) != config[key]:
            raise ValueError(f"Mask backend requires a matching immutable {key}")
    if installed.get("detector_model_id") != config["detector_model_id"]:
        raise ValueError("Detector model identity differs from its configuration")
    packages = installed.get("packages", {})
    if not {"numpy", "torch", "transformers", "SAM-2", "opencv-python"}.issubset(packages):
        raise ValueError("Mask installation must pin required distribution versions")
    if packages["transformers"] != config["transformers_version"]:
        raise ValueError("Detector adapter requires the configured Transformers API version")
    for name, version in packages.items():
        if not isinstance(version, str) or importlib.metadata.version(name) != version:
            raise ValueError(f"Mask dependency version differs from its installation record: {name}")
    files = _pinned(installed.get("files", []), record, environment)
    assets = _pinned(installed.get("assets", []), record)
    detector = project_path(ROOT, config["detector_directory"])
    detector_files = {path.resolve() for path in detector.rglob("*") if path.is_file()}
    if not detector_files or not detector_files.issubset(assets):
        raise ValueError("Every file in the local detector/tokenizer directory must be pinned")
    if project_path(ROOT, config["sam2_checkpoint"]) not in assets:
        raise ValueError("SAM2 checkpoint is missing from the asset inventory")
    for variable, relative in CACHE_DIRECTORIES.items():
        path = project_path(ROOT, relative)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0")
    sys.dont_write_bytecode = True
    record["backend"] = {"packages": packages, "detector_model_id": config["detector_model_id"],
                         "detector_revision": config["detector_revision"], "sam2_source_commit": config["sam2_source_commit"],
                         "sam2_model_revision": config["sam2_model_revision"], "postprocessing": False}
    return config, files


@contextmanager
def detector(prepared):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    import transformers.models.grounding_dino.modeling_grounding_dino as model_source
    import transformers.models.grounding_dino.processing_grounding_dino as processor_source

    config, files = prepared
    if any(Path(module.__file__).resolve() not in files for module in (model_source, processor_source)):
        raise ValueError("Detector implementation is not in the pinned installed-file inventory")
    directory = str(project_path(ROOT, config["detector_directory"]))
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        disable_custom_kernels=True).to("cuda:0").eval()
    try:
        yield model, processor, torch
    finally:
        del model, processor
        torch.cuda.empty_cache()


def detect(runtime, rgb, prompt, request, np):
    model, processor, torch = runtime
    inputs = processor(images=rgb, text=prompt.strip().rstrip(".") + ".", return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        output, inputs.input_ids, threshold=request.minimum_box_score,
        text_threshold=request.minimum_text_score, target_sizes=[rgb.shape[:2]])[0]
    boxes = result["boxes"].detach().cpu().numpy()
    scores = result["scores"].detach().cpu().numpy()
    if (boxes.shape != (len(scores), 4) or not np.isfinite(boxes).all() or not np.isfinite(scores).all() or
            ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Detector produced malformed boxes/scores")
    return np.column_stack((boxes, scores))


@contextmanager
def segmenter(prepared):
    import torch
    import sam2
    import sam2.build_sam as builder
    import sam2.sam2_video_predictor as predictor_source

    config, files = prepared
    configuration = (Path(sam2.__file__).resolve().parent / config["sam2_config"]).resolve()
    if any(Path(module.__file__).resolve() not in files for module in (builder, predictor_source)) or configuration not in files:
        raise ValueError("SAM2 implementation/configuration is outside its installed-file inventory")
    model = builder.build_sam2_video_predictor(config["sam2_config"],
        str(project_path(ROOT, config["sam2_checkpoint"])), device="cuda:0", apply_postprocessing=False)
    try:
        yield model, torch
    finally:
        del model
        torch.cuda.empty_cache()
