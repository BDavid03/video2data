"""Producer request metadata for campaign planning, parsed from source; never imports producers.

Completeness means supplied fields only, not producer validation or permission.
"""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
IDENTITY = "source_split episode_index"


def _fields(names):
    return frozenset((IDENTITY + " " + names).split())


# Keep these source-owned mappings aligned with the named request dataclasses.
REQUEST_FIELDS = {
    "depth": _fields("start_frame stop_frame frame_stride max_image_side num_tokens use_fp16 save_normals gpu_headroom_mib gpu_admission_allowance_mib"),  # GenerateDepthRequest
    "stereo_depth": _fields("start_frame stop_frame max_image_side rectification_run depth_run stereo_run masks_run minimum_disparity num_disparities block_size left_right_tolerance_px minimum_depth_m maximum_depth_m minimum_support_pixels minimum_frame_share holdback_modulus episode_aggregation vertical_tolerance_px reprojection_tolerance_px frame_indices"),  # StereoDepthRequest
    "camera_motion": _fields("depth_run stereo_run anchor_run masks_run opening_frame_count slot0_point_camA_m slot0_point_source feature_budget map_budget minimum_inliers mask_dilation matching_ratio reprojection_limit_px plane_distance_m plane_iterations plane_minimum_points plane_sample_budget plane_max_tilt_degrees lower_image_fraction seed"),  # EstimateCameraRequest
    "object_mesh": _fields("method reconstruction_run scans_only masks_run depth_run anchor_run camera_run pose_run objects cluster_assignments minimum_points voxel_size_m truncation_voxels minimum_fusion_views maximum_grid_points maximum_surface_faces maximum_point_voxels"),  # ObjectMeshRequest
    "object_masks.generate": _fields("depth_run opening_run opening_masks_run keyframe_interval minimum_box_score minimum_text_score minimum_mask_score minimum_mask_pixels maximum_boxes_per_prompt nms_iou assignment_margin maximum_center_distance maximum_identity_gap direction_agreement_iou gpu_headroom_mib gpu_admission_mib"),  # GenerateMasksRequest
    "object_masks.opening_clusters": _fields("depth_run anchor_run camera_run stop_frame minimum_height_m maximum_height_m neighbour_distance_m minimum_cluster_pixels border_margin_pixels"),  # OpeningClustersRequest
    "object_pose": _fields("method mesh_run masks_run depth_run stereo_run anchor_run camera_run hand_run pose_run point_budget mesh_sample_count icp_iterations yaw_steps minimum_points residual_limit_m trim_fraction color_weight_m ambiguity_distance_m ambiguity_color_margin allow_degraded_opening seed minimum_confidence minimum_surface_support recovery_surface_support maximum_translation_step_m maximum_rotation_step_rad hand_mask_dilation"),  # ObjectPoseRequest
    "hand_motion.estimate": _fields("masks_run depth_run minimum_score minimum_mask_iou maximum_detections maximum_track_gap maximum_center_step handedness_penalty assignment_margin crop_expansion gpu_headroom_mib gpu_admission_mib"),  # EstimateHandsRequest
    "hand_motion.kinematics": _fields("estimate_run masks_run depth_run batch_size"),  # KinematicsRequest
    "hand_motion.lift": _fields("kinematics_run masks_run depth_run stereo_run anchor_run camera_run minimum_score minimum_in_frame_fraction minimum_depth_support fit_iterations maximum_depth_shift_m maximum_depth_residual_m outlier_floor_m"),  # LiftHandsRequest
    "scene_assembly": _fields("mode pose_run camera_run mesh_run hand_run parked_hand_run maximum_gap_frames smoothing_radius minimum_rest_frames parking_transition_frames rest_translation_m rest_rotation_rad hand_near_m seating_limit_m contact_distance_m parking_clearance_m supports mass_kg"),  # AssembleSceneRequest
    "retargeting": _fields("scene_run input_run mano_to_robot_scale smooth_reference smooth_reference_window smooth_reference_polyorder surface_project surface_margin surface_granularity adaptive_offset adaptive_offset_alpha adaptive_offset_cap adaptive_offset_smooth_window hand_object_offset ik_position_error_m contact_run uncertainty_run minimum_contact_confidence maximum_randomization_translation_m maximum_randomization_rotation_rad"),  # RetargetRequest
    "policy_training": _fields("retargeting_run compose_run training_run rollout_inspections budget policy_kind seed headroom_mib admission_allowance_mib recipe_variant final_checkpoint_only"),  # PolicyRequest
    "policy_rollout": _fields("training_run checkpoint rollout_run inspection_run scene_run resting_inspection source_root headroom_mib admission_allowance_mib"),  # RolloutRequest
    "campaign": _fields("run artifact_name reason tier target_tier parked_hand_run submission_mesh_method"),  # CampaignRequest
}

REQUEST_SOURCES = {
    "depth": ("depth/generate.py", "GenerateDepthRequest"),
    "stereo_depth": ("stereo_depth/_common.py", "StereoDepthRequest"),
    "camera_motion": ("camera_motion/estimate.py", "EstimateCameraRequest"),
    "object_mesh": ("object_mesh/_common.py", "ObjectMeshRequest"),
    "object_masks.generate": ("object_masks/_generate_common.py", "GenerateMasksRequest"),
    "object_masks.opening_clusters": ("object_masks/opening_clusters.py", "OpeningClustersRequest"),
    "object_pose": ("object_pose/_common.py", "ObjectPoseRequest"),
    "hand_motion.estimate": ("hand_motion/_common.py", "EstimateHandsRequest"),
    "hand_motion.kinematics": ("hand_motion/_common.py", "KinematicsRequest"),
    "hand_motion.lift": ("hand_motion/_common.py", "LiftHandsRequest"),
    "scene_assembly": ("scene_assembly/_common.py", "AssembleSceneRequest"),
    "retargeting": ("retargeting/_common.py", "RetargetRequest"),
    "policy_training": ("policy_training/_common.py", "PolicyRequest"),
    "policy_rollout": ("policy_rollout/_common.py", "RolloutRequest"),
    "campaign": ("campaign/operations.py", "CampaignRequest"),
}

REQUIRED = {
    "depth.generate": (),
    "stereo_depth.rectify": (),
    "stereo_depth.generate": ("depth_run", "rectification_run"),
    "stereo_depth.anchor": ("depth_run", "stereo_run"),
    "camera_motion.estimate": ("depth_run", "stereo_run", "anchor_run"),
    "object_mesh.resolve": (),
    "object_mesh.reconstruct": ("masks_run", "depth_run", "anchor_run", "camera_run"),
    "object_masks.opening_clusters": ("depth_run", "anchor_run", "camera_run"),
    "object_masks.generate": ("depth_run", "opening_run", "opening_masks_run"),
    "object_pose.initialise": ("mesh_run", "masks_run", "depth_run", "anchor_run", "camera_run"),
    "object_pose.track": ("pose_run", "mesh_run", "masks_run", "depth_run", "stereo_run", "anchor_run", "camera_run"),
    "object_pose.compose": ("pose_run", "camera_run"),
    "hand_motion.estimate": ("masks_run", "depth_run"),
    "hand_motion.kinematics": ("estimate_run", "masks_run", "depth_run"),
    "hand_motion.lift": ("kinematics_run", "masks_run", "depth_run", "stereo_run", "anchor_run", "camera_run"),
    "scene_assembly.assemble": ("pose_run", "camera_run", "mesh_run", "parked_hand_run", "mass_kg", "supports"),
    "retargeting.export": ("scene_run",),
    "retargeting.assets": ("input_run",),
    "retargeting.retarget": ("input_run",),
    "retargeting.support": ("input_run",),
    "retargeting.inspect": ("input_run",),
    "policy_training.compose": ("retargeting_run", "policy_kind", "budget"),
    "policy_training.train": ("compose_run", "policy_kind", "budget"),
    "policy_training.select": ("training_run", "rollout_inspections", "policy_kind", "budget"),
    "policy_rollout.rollout": ("training_run", "checkpoint"),
    "policy_rollout.inspect": ("rollout_run",),
    "policy_rollout.self_score": ("inspection_run", "scene_run"),
    "campaign.pin": ("run", "artifact_name", "reason"),
}


def request_fields(operation):
    """Full actual-action dataclass keys, including keys omitted by templates."""
    if operation not in REQUIRED:
        raise ValueError(f"No source-owned request contract for {operation}")
    feature = operation.split(".", 1)[0]
    return sorted(REQUEST_FIELDS.get(operation, REQUEST_FIELDS.get(feature, ())))


def _missing(value):
    if value is None or value == "":
        return True
    if isinstance(value, dict):
        return not value or any(_missing(item) for item in value.values())
    if isinstance(value, list):
        return not value or any(_missing(item) for item in value)
    return isinstance(value, str) and not value.strip()


def unresolved_fields(operation, draft):
    """Missing required field names.

    Optional nulls (stop_frame, source_root, resting_inspection, resting hand_run, scan
    reconstruction_run, pin tier) are allowed; empty objects selects the whole episode and
    empty cluster_assignments allows geometric assignment.
    """
    allowed = request_fields(operation)
    unknown = sorted(set(draft) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {operation} request fields: {', '.join(unknown)}")
    required = [*IDENTITY.split(), *REQUIRED[operation]]
    if operation == "camera_motion.estimate" and draft.get("opening_frame_count", 20) is None:
        required.append("masks_run")
    if operation == "camera_motion.estimate" and draft.get("slot0_point_camA_m") is not None:
        required.append("slot0_point_source")
    if operation == "scene_assembly.assemble" and draft.get("mode", "resting") == "full":
        required.append("hand_run")
    return sorted(name for name in set(required) if _missing(draft.get(name)))


def _dataclass_fields(raw, class_name):
    """Annotated field names of one non-inherited class, parsed from source without importing it."""
    classes = [node for node in ast.parse(raw.decode("utf-8-sig")).body
               if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1 or classes[0].bases:
        raise ValueError("Request metadata requires one explicit, non-inherited dataclass")
    return {node.target.id for node in classes[0].body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}


def request_draft(operation, source_split, episode_index, parameters=None, bindings=None, root=None):
    """Merge a producer's request template with explicit planner values; reads source/JSON, writes nothing."""
    fields = request_fields(operation)
    base = Path(root) if root is not None else ROOT
    feature, suboperation = operation.split(".", 1)
    source_name, class_name = REQUEST_SOURCES.get(operation, REQUEST_SOURCES.get(feature))
    source_path = f"track3/features/{source_name}"
    raw = base.joinpath(source_path).read_bytes()
    if _dataclass_fields(raw, class_name) != set(fields):
        raise ValueError(f"Request field catalog differs from {class_name}; update the campaign contract")
    cli_operation = suboperation.replace("_", "-")
    template = f"track3/features/{feature}/{cli_operation}.json"
    draft = json.loads(base.joinpath(template).read_text(encoding="utf-8-sig"))
    parameters, bindings = parameters or {}, bindings or {}
    for values in (draft, parameters, bindings):
        unknown = sorted(set(values) - set(fields))
        if unknown:
            raise ValueError(f"Unknown {operation} request fields: {', '.join(unknown)}")
    draft.update(parameters)
    draft.update(bindings)
    draft.update(source_split=source_split, episode_index=episode_index)
    # A candidate pin records a role; only campaign promotion explicitly sets tier.
    if operation == "campaign.pin" and "tier" not in parameters:
        draft["tier"] = None
    missing_bindings = sorted(name for name, value in bindings.items() if _missing(value))
    resolved_bindings = {name: value for name, value in bindings.items() if name not in missing_bindings}
    if operation == "campaign.pin" and not _missing(draft.get("run")):
        resolved_bindings["run"] = draft["run"]
    required_fields = unresolved_fields(operation, draft)
    missing_choices = sorted(set(required_fields) - set(missing_bindings))
    module = f"track3.features.{feature}"
    cli = None if feature == "depth" else cli_operation
    argv = ["python", "-m", module, *([cli] if cli else []), "--request", "{request_file}"]
    return {"request_draft": draft, "request_fields": fields,
            "request_source": source_path, "request_source_class": class_name,
            "request_source_sha256": hashlib.sha256(raw).hexdigest(),
            "request_template": template, "request_module": module,
            "cli_operation": cli, "command_argv": argv,
            "resolved_bindings": resolved_bindings,
            "required_fields": required_fields, "missing_bindings": missing_bindings,
            "missing_choices": missing_choices,
            "request_complete": not missing_bindings and not missing_choices}
