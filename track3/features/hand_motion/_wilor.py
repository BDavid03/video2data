"""Optional pinned WiLoR integration, loaded only in an approved local environment."""

from contextlib import contextmanager
import importlib.metadata
import os
from pathlib import Path
import sys

from track3.paths import project_path
from ._common import FEATURE, ROOT, read_json, record_input

SOURCE_COMMIT = "ebec42f94c389070cdd7dda6fd1bf0b4a659c960"


def provisioned(record):
    record_input(FEATURE / "backends.json", record)
    config = read_json(FEATURE / "backends.json")["wilor"]
    environment = project_path(ROOT, config["environment"])
    if Path(sys.prefix).resolve() != environment:
        raise PermissionError("Run estimation only in the isolated project WiLoR environment")
    if not config.get("installation_record") or not config.get("installation_record_sha256"):
        raise PermissionError("WiLoR has not been approved and provisioned")
    path = project_path(ROOT, config["installation_record"])
    if record_input(path, record)["sha256"] != config["installation_record_sha256"]:
        raise ValueError("WiLoR installation record differs from its pin")
    installation = read_json(path)
    if (installation.get("status") != "completed" or installation.get("user_approved") is not True or
            installation.get("backend") != "wilor-mini" or installation.get("source_commit") != SOURCE_COMMIT or
            config["source_commit"] != SOURCE_COMMIT or
            project_path(ROOT, installation["environment"]) != environment or
            installation.get("licensed_assets_acknowledged") is not True):
        raise PermissionError("An approved pinned WiLoR installation and licensed MANO asset are required")
    packages = installation.get("packages", {})
    if not {"numpy", "torch", "wilor-mini", "ultralytics", "opencv-python"}.issubset(packages):
        raise ValueError("WiLoR installation must pin required distribution versions")
    for name, version in packages.items():
        if not isinstance(version, str) or not version or importlib.metadata.version(name) != version:
            raise ValueError(f"WiLoR package differs from its recorded version: {name}")
    files = {}
    for item in installation.get("files", []):
        path = project_path(ROOT, item["path"])
        if not path.is_relative_to(environment) or path in files:
            raise ValueError("WiLoR installed files must be unique and inside its environment")
        if record_input(path, record)["sha256"] != item["sha256"]:
            raise ValueError("WiLoR installed source differs from its installation record")
        files[path] = item["sha256"]
    if not files:
        raise ValueError("WiLoR needs an explicit installed-file inventory")
    weights = project_path(ROOT, config["weights_root"])
    assets = installation.get("assets", {})
    for name in ("mano_mean_params.npz", "MANO_RIGHT.pkl", "wilor_final.ckpt", "detector.pt"):
        if assets.get(name) != record_input(weights / "pretrained_models" / name, record):
            raise ValueError(f"WiLoR asset is missing or differs from its pin: {name}")
    for variable, relative in {
        "TORCH_HOME": ".cache/hand-motion/torch", "XDG_CACHE_HOME": ".cache/hand-motion",
        "HF_HOME": ".cache/hand-motion/huggingface", "YOLO_CONFIG_DIR": ".cache/hand-motion/ultralytics",
        "MPLCONFIGDIR": ".cache/hand-motion/matplotlib", "TMPDIR": ".runtime/tmp/hand-motion",
        "TMP": ".runtime/tmp/hand-motion", "TEMP": ".runtime/tmp/hand-motion",
    }.items():
        path = project_path(ROOT, relative)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", YOLO_AUTOINSTALL="false",
                      WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0")
    sys.dont_write_bytecode = True
    record["backend"] = {"name": "wilor-mini", "source_commit": SOURCE_COMMIT,
                         "packages": packages, "assets": assets, "device": "cuda:0"}
    return weights, files


@contextmanager
def pipeline(record, prepared, focal, image_shape):
    import torch
    import wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline as module
    import wilor_mini.models.wilor as model_module

    weights, files = prepared
    for imported in (module, model_module):
        if Path(imported.__file__).resolve() not in files:
            raise ValueError("Imported WiLoR module is outside the pinned installed-file inventory")
    if not torch.cuda.is_available():
        raise RuntimeError("WiLoR estimation requires the authorized CUDA runtime")
    original_download = module.hf_hub_download

    def no_download(*args, **kwargs):
        raise PermissionError("Implicit WiLoR downloads are disabled; provision the exact assets separately")

    module.hf_hub_download = no_download
    model = None
    try:
        model = module.WiLorHandPose3dEstimationPipeline(
            device=torch.device("cuda:0"), dtype=torch.float32, verbose=False,
            wilor_pretrained_dir=str(weights), focal_length=float(focal * 256 / max(image_shape)))
        record["backend"]["dtype"] = "float32"
        yield model
    finally:
        module.hf_hub_download = original_download
        del model


def detections(model, bgr, request, np):
    output = model.hand_detector(bgr, conf=request.minimum_score, max_det=request.maximum_detections,
                                 verbose=False)[0]
    rows = output.boxes.data.detach().cpu().numpy()
    if rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all():
        raise ValueError("Detector must return finite xyxy, confidence and handedness rows")
    if ((rows[:, 4] < 0) | (rows[:, 4] > 1)).any() or not np.isin(rows[:, 5], [0, 1]).all():
        raise ValueError("Detector returned invalid confidence or handedness class")
    if (rows[:, 2:4] <= rows[:, :2]).any():
        raise ValueError("Detector returned a degenerate hand box")
    rows = rows[rows[:, 4] >= request.minimum_score]
    return rows[np.argsort(-rows[:, 4], kind="stable")][:request.maximum_detections]


def regress(model, image, detection, side, request, np):
    output = model.predict_with_bboxes(image, bboxes=np.asarray([detection[:4]], dtype=np.float32),
                                       is_rights=np.asarray([side], dtype=np.int32),
                                       rescale_factor=request.crop_expansion)
    if len(output) != 1 or int(output[0]["is_right"]) != side:
        raise ValueError("WiLoR did not preserve the explicitly assigned anatomical side")
    prediction = output[0]["wilor_preds"]
    values = {"wrist_orient_camA": np.asarray(prediction["global_orient"]).reshape(-1),
              "finger_pose": np.asarray(prediction["hand_pose"]).reshape(-1),
              "translation_camA_m": np.asarray(prediction["pred_cam_t_full"]).reshape(-1),
              "betas": np.asarray(prediction["betas"]).reshape(-1)}
    for key, size in (("wrist_orient_camA", 3), ("finger_pose", 45), ("translation_camA_m", 3), ("betas", 10)):
        if values[key].shape != (size,) or not np.isfinite(values[key]).all():
            raise ValueError(f"Malformed WiLoR parameter: {key}")
    if values["translation_camA_m"][2] <= 0:
        raise ValueError("WiLoR estimated a model origin behind the camera")
    # This pinned pipeline already converts both left-hand axial rotations.
    # Store anatomical parameters unchanged; downstream kinematics must not mirror again.
    return values
