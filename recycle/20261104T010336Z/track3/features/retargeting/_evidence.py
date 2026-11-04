"""Support admission for research sidecars; no simulator or sampling backend."""

import math


def contact_admission(evidence, request, frame_indices, pts, object_names):
    """Admit only explicitly supported intervals on the exact source clock."""
    if frame_indices != list(range(len(pts))):
        raise ValueError("Pinned retargeting requires a complete zero-origin source-frame grid")
    if (evidence.get("schema_version") != 1 or evidence.get("source_split") != request.source_split
            or evidence.get("episode_index") != request.episode_index
            or evidence.get("frame_index") != frame_indices or evidence.get("pts") != pts
            or evidence.get("time_base") != "1/10240"):
        raise ValueError("Contact intervals must match the scene's complete source frame/PTS clock")
    intervals = evidence.get("intervals")
    if not isinstance(intervals, list):
        raise ValueError("Contact evidence requires explicit intervals")
    allowed, denied = set(), set()
    decisions = []
    for interval in intervals:
        start, end = interval.get("start_frame"), interval.get("end_frame")
        side, object_name = interval.get("side"), interval.get("object_name")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(pts)
                or side not in ("left", "right") or object_name not in object_names
                or interval.get("start_pts") != pts[start] or interval.get("end_pts") != pts[end - 1]):
            raise ValueError("Malformed or off-clock contact interval")
        confidence = interval.get("confidence")
        support = interval.get("support", {})
        ambiguity = interval.get("ambiguity")
        if not isinstance(support, dict) or not isinstance(ambiguity, list):
            raise ValueError("Contact support and ambiguity must be explicit")
        reasons = []
        if interval.get("accepted") is not True or interval.get("physical_contact_confirmed") is not True:
            reasons.append("physical_contact_unconfirmed")
        if interval.get("phase") not in ("contact", "grasp", "lift"):
            reasons.append("phase_does_not_establish_hand_object_contact")
        if ambiguity:
            reasons.append("ambiguous_interval")
        if support.get("contact_supported") is not True:
            reasons.append("missing_independent_contact_support")
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not request.minimum_contact_confidence <= confidence <= 1):
            reasons.append("missing_or_insufficient_confidence")
        accepted = not reasons
        decisions.append({"interval": interval, "admitted": accepted, "reasons": reasons,
                          "confidence": confidence, "confidence_is_calibrated": False})
        destination = allowed if accepted else denied
        destination.update((frame, ("left", "right").index(side), object_names.index(object_name) + 1)
                           for frame in range(start, end))
    return allowed - denied, {"schema_version": 1, "time_base": "1/10240", "frame_index": frame_indices,
                     "pts": pts, "end_frame_convention": "exclusive", "end_pts_convention": "last_included_sample",
                     "decisions": decisions, "physical_feasibility_established": False,
                     "unsupported_contact_policy": "zero_position_normal_and_body_id"}


def randomization_proposal(evidence, request):
    """Empirical extrema of independently measured scene errors; components over their cap are rejected, not clipped."""
    if (evidence.get("schema_version") != 1 or evidence.get("provenance") != "video_only"
            or evidence.get("split") != request.source_split
            or evidence.get("episode_index") != request.episode_index
            or evidence.get("evidence_kind") != "independent_measured_scene_errors"
            or evidence.get("coordinate_frame") != "world"
            or evidence.get("error_definition") != "prediction_minus_independent_measurement"):
        raise ValueError("Randomization proposals require episode-matched independent measured scene errors")
    components = evidence.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Measured scene error components are required")
    limits = {"translation_error": ("m", request.maximum_randomization_translation_m),
              "rotation_error": ("rad", request.maximum_randomization_rotation_rad)}
    proposals, rejected, seen = [], [], set()
    for item in components:
        quantity, axis, object_name = item.get("quantity"), item.get("axis"), item.get("object_name")
        identity = (quantity, axis, object_name)
        if identity in seen or quantity not in limits or axis not in ("x", "y", "z") or not isinstance(object_name, str) or not object_name:
            raise ValueError("Error components require unique object/quantity/axis identities")
        seen.add(identity)
        units, cap = limits[quantity]
        samples, support = item.get("samples"), item.get("support")
        if (item.get("units") != units or not isinstance(samples, list) or len(samples) < 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in samples)
                or not isinstance(support, list) or len(support) != len(samples)):
            raise ValueError("Each measured error needs finite signed samples, physical units and matching support")
        keys = set()
        for measurement in support:
            if (not isinstance(measurement, dict) or type(measurement.get("frame_index")) is not int
                    or measurement["frame_index"] < 0 or type(measurement.get("pts")) is not int
                    or not isinstance(measurement.get("source_artifact"), dict)
                    or set(measurement["source_artifact"]) != {"path", "sha256"}
                    or not isinstance(measurement.get("method"), str) or not measurement["method"]):
                raise ValueError("Error support must identify frame, PTS, measurement method and hashed source artifact")
            key = (measurement["frame_index"], measurement["pts"], measurement["source_artifact"]["path"])
            if key in keys:
                raise ValueError("Duplicate scene-error support is refused")
            keys.add(key)
        proposal = {"object_name": object_name, "quantity": quantity, "axis": axis, "units": units,
                    "empirical_lower": min(samples), "empirical_upper": max(samples),
                    "sample_count": len(samples), "support": support, "maximum_absolute_bound": cap,
                    "distribution": None, "sampling_enabled": False}
        if max(abs(min(samples)), abs(max(samples))) > cap:
            proposal["reason"] = "measured_error_exceeds_predeclared_cap_reconstruction_review_required"
            rejected.append(proposal)
        else:
            proposals.append(proposal)
    return {"schema_version": 1, "status": "proposal_only", "components": proposals, "rejected": rejected,
            "calibrated_uncertainty": False, "sampling_enabled": False,
            "limitations": ["Empirical extrema do not establish population coverage or a sampling distribution.",
                            "No friction, mass, collision or policy parameter is inferred from pose errors."]}
