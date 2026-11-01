"""Admission of final kit outputs, called only by an authorized packing operation."""

import csv
import math
import re

from track3.provenance import input_file, sha256_file
from ._common import ROOT, KIT, artifact, read_json

POSE_COLUMNS = ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")
POSE_SOURCE = ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y",
               "quaternion_z", "quaternion_w")
PROVENANCE_COLUMNS = ("eval_script_sha256", "eval_code_sha256", "checkpoint_sha256",
                      "packer_script_sha256", "uploader_script_sha256", "code_commit_url")


def _csv(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(columns):
            raise ValueError("Packed pose columns differ from the pinned kit contract")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Packed CSV contains a malformed row")
    return rows


def _exact_roster(rows, roster):
    keys = [row["row_id"] for row in rows]
    if (not keys or len(set(roster)) != len(roster) or len(set(keys)) != len(keys)
            or set(keys) != set(roster)):
        raise ValueError("Packed output must contain exactly the sample's unique row IDs")


def _poses(path, parent, document, record, pq):
    rows = _csv(path, ("row_id", *POSE_COLUMNS, *PROVENANCE_COLUMNS))
    with (KIT / "data/track_3_sample_submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        roster = [row["row_id"] for row in csv.DictReader(stream)]
    _exact_roster(rows, roster)
    expected = {}
    uniform = {
        "packer_script_sha256": sha256_file(KIT / "tools/pack_submission.py"),
        "uploader_script_sha256": sha256_file(KIT / "submit/v2d_submit.py"),
        "code_commit_url": document["code_commit_url"],
    }
    for episode, selected in document["rollouts"].items():
        staged = artifact(parent, f"results/episode_{int(episode):06d}.parquet", record)
        table = pq.ParquetFile(staged).read()
        metadata = {key.decode(): value.decode() for key, value in (table.schema.metadata or {}).items()}
        provenance = {**uniform, "checkpoint_sha256": selected["checkpoint"]["sha256"]}
        for name in ("eval_code_sha256", "eval_script_sha256"):
            provenance[name] = metadata["flash_chord." + name]
            if name in uniform and uniform[name] != provenance[name]:
                raise ValueError("Packed poses mix evaluator builds")
            uniform[name] = provenance[name]
        for row in table.to_pylist():
            key = f"track_3/e{int(episode):06d}/w{row['environment_id']:03d}/f{row['reference_frame']:06d}/o{row['object_body_id']}"
            if key in expected:
                raise ValueError("Staged trajectory contains duplicate pose rows")
            expected[key] = ([float(row["achieved_" + name]) for name in POSE_SOURCE], provenance)
    if set(expected) != set(roster):
        raise ValueError("Staged trajectories differ from the packed roster")
    for row in rows:
        values, provenance = expected[row["row_id"]]
        if any(row[name] != value for name, value in provenance.items()):
            raise ValueError("Packed pose provenance differs from the selected input or kit")
        actual = [float(row[name]) for name in POSE_COLUMNS]
        # The pinned Track 3 packer copies achieved poses and writes %.6f;
        # it performs no frame conversion or quaternion normalization here.
        if (not all(math.isfinite(value) for value in actual)
                or actual != [float(format(value, ".6f")) for value in values]
                or math.sqrt(sum(value * value for value in actual[3:])) < 1e-8):
            raise ValueError("Packed pose differs from the selected rollout or is invalid after rounding")
    return {"row_count": len(rows), "episode_count": len(document["rollouts"]),
            "pose_comparison": "Exact numeric equality after six-decimal formatting of staged achieved poses"}


def _meshes(path, document, pq):
    import numpy as np

    table = pq.ParquetFile(path).read()
    if table.column_names != ["row_id", "x", "y", "z", "code_commit_url"]:
        raise ValueError("Packed mesh columns differ from the pinned kit contract")
    rows = table.to_pylist()
    roster = pq.ParquetFile(KIT / "data/track_3_mesh_sample_submission.parquet").read(
        columns=["row_id"]).column(0).to_pylist()
    _exact_roster(rows, roster)
    groups = {}
    for row in rows:
        match = re.fullmatch(r"t3m_(\d{6})_(.+)_([01])_(\d{6})", row["row_id"])
        if match is None or row["code_commit_url"] != document["code_commit_url"]:
            raise ValueError("Packed mesh identity or commit differs from the plan")
        episode, name, array, point = match.groups()
        block = groups.setdefault((int(episode), name), {0: {}, 1: {}})[int(array)]
        block[int(point)] = [row[axis] for axis in ("x", "y", "z")]
    if set(groups) != {(row["episode_index"], row["name"]) for row in document["meshes"]}:
        raise ValueError("Packed mesh pairs differ from the plan")
    budgets = []
    for (episode, name), blocks in sorted(groups.items()):
        arrays = []
        for block in (blocks[0], blocks[1]):
            if not block or set(block) != set(range(len(block))):
                raise ValueError("Packed mesh array indices must be contiguous from zero")
            values = np.asarray([block[index] for index in range(len(block))], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Packed mesh contains nonfinite values")
            arrays.append(values)
        vertices, faces = arrays
        if (not np.equal(faces, np.floor(faces)).all() or (faces < 0).any()
                or (faces >= len(vertices)).any()):
            raise ValueError("Packed mesh faces must index the packed vertex array")
        triangles = vertices[faces.astype(np.int64)]
        double_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                              triangles[:, 2] - triangles[:, 0]), axis=1)
        if not np.isfinite(double_area).all() or not (double_area > 0).any():
            raise ValueError("Packed mesh has no finite positive-area surface")
        budgets.append({"episode_index": episode, "name": name, "vertex_rows": len(vertices),
                        "face_rows": len(faces), "positive_area_faces": int((double_area > 0).sum())})
    return {"row_count": len(rows), "meshes": budgets,
            "geometry_scope": "Roster, finite coordinates, integer in-range faces and positive surface area; no shape accuracy claim"}


def admit_packed(kind, path, parent, record):
    import pyarrow.parquet as pq

    identity = input_file(ROOT, path)
    document = read_json(artifact(parent, "submission.json", record))
    if kind == "poses":
        result = _poses(path, parent, document, record, pq)
    elif kind == "meshes":
        result = _meshes(path, document, pq)
    else:
        raise ValueError("Unknown packed submission kind")
    return {"schema_version": 1, "kind": kind, "blocking": False, "file": identity,
            "submission_run": input_file(ROOT, parent[0]), **result}
