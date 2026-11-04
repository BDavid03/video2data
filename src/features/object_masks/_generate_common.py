"""Full-video mask request, authorization and run lifecycle."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

from src.paths import output_path
from src.provenance import run_fields
from src.runs import ROOT, dataset_revision, new_run_id, read_json, recorded, same_revision
from .opening_clusters import FEATURE, _authorize, _names_source


@dataclass(frozen=True)
class GenerateMasksRequest:
    source_split: str = "public"
    episode_index: int = 39
    depth_run: str | None = None
    opening_run: str | None = None
    opening_masks_run: str | None = None
    keyframe_interval: int = 10
    minimum_box_score: float = 0.3
    minimum_text_score: float = 0.25
    minimum_mask_score: float = 0.7
    minimum_mask_pixels: int = 50
    maximum_boxes_per_prompt: int = 8
    nms_iou: float = 0.7
    assignment_margin: float = 0.05
    maximum_center_distance: float = 0.35
    maximum_identity_gap: int = 40
    direction_agreement_iou: float = 0.3
    gpu_headroom_mib: int = 2048
    gpu_admission_mib: int = 4096

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation") or type(self.episode_index) is not int or self.episode_index < 0:
            raise ValueError("Mask generation requires a named public/evaluation episode")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")
        if (self.opening_run is None) != (self.opening_masks_run is None):
            raise ValueError("Opening identity requires both pose and cluster-mask runs")
        for key in ("keyframe_interval", "minimum_mask_pixels", "maximum_boxes_per_prompt",
                    "maximum_identity_gap", "gpu_headroom_mib", "gpu_admission_mib"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid mask generation integer: {key}")
        if self.keyframe_interval > 60 or self.maximum_boxes_per_prompt > 16:
            raise ValueError("Mask windows are bounded to 61 frames and 16 boxes per prompt")
        for key in ("minimum_box_score", "minimum_text_score", "minimum_mask_score", "nms_iou",
                    "assignment_margin", "maximum_center_distance", "direction_agreement_iou"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Invalid mask fraction: {key}")


def authorize(request):
    _authorize(request, "generate_authorized", "mask generation")


@contextmanager
def operation_run(request, parents):
    authorize(request)
    revision = same_revision(parents, dataset_revision())
    record = run_fields(ROOT, "object_masks", "generate", asdict(request),
                        parents=[path for path, _ in parents], allowed=("video_only",),
                        inputs=[FEATURE / "runtime.json", FEATURE / "backends.json", FEATURE / "prompts.json",
                                "data/configs/splits/public-v1.json", _names_source(request), "data/reports/data-audit/videos.json"])
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, artifacts={}, frames=[], gpu=None, method="grounding_dino_sam2")
    with recorded(output_path(ROOT, f"data/outputs/object_masks/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as result:
        yield result
