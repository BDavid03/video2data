"""Optional, producer-owned image evidence for inspection; no estimation or fitting."""

import json
import math

from track3.paths import project_path
from track3.provenance import input_file


def _point(value, width, height):
    return (isinstance(value, list) and len(value) == 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in value)
            and 0 <= value[0] < width and 0 <= value[1] < height)


VISIBILITY = ("observed", "occluded", "unknown")
SUPPORT = ("admitted", "rejected", "unknown")
CHANNELS = ("tracks", "reprojection", "states", "surfaces")


def _check_tracks(tracks, width, height):
    identities = set()
    for track in tracks:
        if (set(track) != {"id", "xy", "visibility"} or not isinstance(track["id"], str)
                or not track["id"] or track["id"] in identities
                or track["visibility"] not in VISIBILITY
                or (track["xy"] is not None and not _point(track["xy"], width, height))
                or (track["visibility"] == "observed" and track["xy"] is None)):
            raise ValueError("Invalid track identity, calibrated pixel or visibility")
        identities.add(track["id"])


def _check_reprojection(residuals, width, height):
    for residual in residuals:
        if (set(residual) != {"id", "observed_xy", "predicted_xy", "support"}
                or not isinstance(residual["id"], str)
                or not _point(residual["observed_xy"], width, height)
                or not _point(residual["predicted_xy"], width, height)
                or residual["support"] not in SUPPORT):
            raise ValueError("Reprojection endpoints/support must be explicit on the shared grid")


def _check_states(states):
    for state in states:
        if (set(state) != {"identity", "visibility", "support", "discontinuity", "uncertainty"}
                or not isinstance(state["identity"], str)
                or state["visibility"] not in VISIBILITY or state["support"] not in SUPPORT
                or state["discontinuity"] not in ("none", "lost", "reacquired", "identity_switch", "unknown")):
            raise ValueError("Invalid explicit visibility/support/discontinuity state")
        uncertainty = state["uncertainty"]
        if uncertainty is not None and (
                set(uncertainty) != {"kind", "value", "units"}
                or uncertainty["kind"] not in ("member_disagreement", "uncalibrated_score")
                or type(uncertainty["value"]) not in (int, float)
                or not math.isfinite(uncertainty["value"]) or uncertainty["value"] < 0
                or not isinstance(uncertainty["units"], str) or not uncertainty["units"]):
            raise ValueError("Only explicitly uncalibrated uncertainty is admitted; null means missing")


def _check_surfaces(surfaces):
    for surface in surfaces:
        if (set(surface) != {"identity", "geometry", "status", "evidence"}
                or not isinstance(surface["identity"], str)
                or surface["geometry"] not in ("visual", "collision")
                or surface["status"] not in ("observed", "inferred", "mixed", "unknown")
                or not isinstance(surface["evidence"], str) or not surface["evidence"].strip()):
            raise ValueError("Surface provenance needs a producer evidence explanation")


def admit(root, value, sources, grid, pts, selected):
    """Admit an annotation file that is a hashed artifact of exactly one admitted producer."""
    path = project_path(root, value)
    entry = input_file(root, path)
    owners = [run_path for run_path, run in sources.values()
              if path.is_relative_to(run_path.parent)
              and run.get("artifacts", {}).get(path.relative_to(run_path.parent).as_posix()) == entry["sha256"]]
    if len(owners) != 1:
        raise ValueError("Inspection annotations require one exact admitted producer artifact owner")
    producer = input_file(root, owners[0])
    if path.stat().st_size > 8_000_000:
        raise ValueError("Inspection annotations exceed the 8 MB budget")
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if (set(document) != {"schema_version", "grid", "producer", "image_size", "frames"} or document["schema_version"] != 1
            or document["grid"] != grid or document["producer"] != producer["path"]):
        raise ValueError("Annotation identity/schema differs from the selected producer and grid")
    size = document["image_size"]
    if (not isinstance(size, list) or len(size) != 2
            or any(type(n) is not int or not 1 <= n <= 16384 for n in size)):
        raise ValueError("Declare annotation image_size as [width, height]")
    width, height = size
    rows = document["frames"]
    if not isinstance(rows, list) or len(rows) > 256:
        raise ValueError("Annotations admit at most 256 explicit source frames")
    indexed = {}
    for row in rows:
        if set(row) != {"frame_index", "source_pts", *CHANNELS}:
            raise ValueError("Annotation channels must be explicit, including empty channels")
        i = row["frame_index"]
        if type(i) is not int or i not in selected or i in indexed or row["source_pts"] != pts[i]:
            raise ValueError("Annotation frame/PTS differs, is duplicated or was not selected")
        if any(not isinstance(row[name], list) or len(row[name]) > 2048 for name in CHANNELS):
            raise ValueError("Each annotation channel is bounded to 2048 observations")
        _check_tracks(row["tracks"], width, height)
        _check_reprojection(row["reprojection"], width, height)
        _check_states(row["states"])
        _check_surfaces(row["surfaces"])
        indexed[i] = row
    return entry, {"image_size": size, "producer": producer, "frames": indexed}


def overlay(image, row, image_size, cv2):
    """Draw observed tracks and admitted/rejected residuals without interpolation."""
    if [image.shape[1], image.shape[0]] != image_size:
        raise ValueError("Annotation grid differs from saved RGB dimensions")
    pixels = image.copy()
    for track in row["tracks"]:
        if track["visibility"] == "observed":
            xy = tuple(round(x) for x in track["xy"])
            cv2.circle(pixels, xy, 3, (30, 240, 130), 1)
            cv2.putText(pixels, track["id"][:32], xy, cv2.FONT_HERSHEY_SIMPLEX, .35, (30, 240, 130), 1)
    for residual in row["reprojection"]:
        color = {"admitted": (255, 190, 20), "rejected": (255, 40, 80), "unknown": (150, 150, 150)}[residual["support"]]
        cv2.arrowedLine(pixels, tuple(round(x) for x in residual["predicted_xy"]),
                       tuple(round(x) for x in residual["observed_xy"]), color, 1, tipLength=.2)
    return pixels
