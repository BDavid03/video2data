"""Public reconstruction comparisons with explicit reference-scan coverage."""

from track3.provenance import input_file
from .diagnostics import ROOT, _artifact, _artifact_entry, _consume, _json, _path

UNSCANNED = "No wooden_spoon reference scan in the pinned public release"


def _objects(parent, episode, names, record):
    """Map object name to (path, metadata, input entry) for same-episode reconstructed meshes."""
    document = _json(_artifact(parent, "mesh.json", record))
    objects = {}
    for item in document["objects"]:
        name = item["name"]
        if (name in objects or name not in names or item["source"] != "reconstructed"
                or item["source_split"] != "public" or item["source_episode"] != episode):
            raise ValueError("Mesh diagnostic needs unique same-episode public reconstructed objects")
        entry = _artifact_entry(parent, item["file"], record)
        if entry["sha256"] != item["sha256"]:
            raise ValueError("Mesh metadata hash differs")
        objects[name] = (_path(entry["path"]), item, entry)
    if not objects:
        raise ValueError("No reconstructed meshes to score")
    return objects


def _chamfer(reference, candidate_path, seed, metric, np, trimesh):
    candidate = trimesh.load(str(candidate_path), force="mesh", process=False)
    value = float(metric.object_chamfer_cm(
        (np.asarray(reference.vertices), np.asarray(reference.faces)),
        (np.asarray(candidate.vertices), np.asarray(candidate.faces)), seed=seed))
    if not np.isfinite(value) or value < 0:
        raise ValueError("Mesh metric did not produce a finite nonnegative distance")
    return value


def _mean(rows, key, np):
    return float(np.mean([row[key] for row in rows])) if rows else None


def compare_meshes(parent, baseline, episode, names, snapshot, record, metric, sampler, np, trimesh):
    candidates = _objects(parent, episode, names, record)
    baselines = _objects(baseline, episode, names, record) if baseline else None
    if baselines is not None and set(candidates) != set(baselines):
        raise ValueError("Mesh comparison requires the same reconstructed object set in both runs")
    rows = []
    for name in names:
        if name not in candidates:
            continue
        path, item, entry = candidates[name]
        row = {"name": name, "candidate": entry, "method": item["method"]}
        if baseline:
            baseline_path, baseline_item, baseline_entry = baselines[name]
            row.update(baseline=baseline_entry, baseline_method=baseline_item["method"])
        if name == "wooden_spoon":
            # A declared gap in the pinned release; an absent scan for any scanned object still fails.
            row.update(status="unscorable", cd_o_cm=None, reason=UNSCANNED)
            if baseline:
                row.update(baseline_cd_o_cm=None, delta_cd_o_cm=None)
            rows.append(row)
            continue
        reference_path = _consume(sampler.mesh_path_for(snapshot / "track_3/public/mesh", name), record)
        reference = trimesh.load(str(reference_path), force="mesh", process=False)
        seed = metric.object_seed(episode, names.index(name))
        value = _chamfer(reference, path, seed, metric, np, trimesh)
        row.update(status="scored", cd_o_cm=value, reference=input_file(ROOT, reference_path))
        if baseline:
            baseline_value = _chamfer(reference, baseline_path, seed, metric, np, trimesh)
            row.update(baseline_cd_o_cm=baseline_value, delta_cd_o_cm=value - baseline_value)
        rows.append(row)
    scored = [row for row in rows if row["status"] == "scored"]
    result = {"objects": rows, "status": "scored" if len(scored) == len(rows) else
              ("partially_scored" if scored else "unscorable"),
              "candidate_run": input_file(ROOT, parent[0]),
              "requested_object_count": len(rows), "scored_object_count": len(scored),
              "unscorable_object_count": len(rows) - len(scored),
              "reference_scan_coverage": len(scored) / len(rows),
              "cd_o_cm": _mean(scored, "cd_o_cm", np),
              "geometry_scope": "Original reconstructed and reference meshes; kit metric, without submission-budget decimation"}
    if baseline:
        result.update(baseline_run=input_file(ROOT, baseline[0]),
                      baseline_cd_o_cm=_mean(scored, "baseline_cd_o_cm", np),
                      delta_cd_o_cm=_mean(scored, "delta_cd_o_cm", np),
                      delta_convention="Candidate minus baseline; negative means lower distance on scored objects")
    return result


def mesh_means(rows, np):
    scored = [row for row in rows if row["scored_object_count"] > 0]
    result = {"cd_o_cm": _mean(scored, "cd_o_cm", np),
              "episode_count": len(rows), "scored_episode_count": len(scored),
              "scored_object_count": sum(row["scored_object_count"] for row in rows),
              "unscorable_object_count": sum(row["unscorable_object_count"] for row in rows),
              "aggregation": "Equal weight per scored episode; unavailable reference scans are excluded, never zero"}
    if any("baseline_run" in row for row in rows):
        result.update(baseline_cd_o_cm=_mean(scored, "baseline_cd_o_cm", np),
                      delta_cd_o_cm=_mean(scored, "delta_cd_o_cm", np))
    return result
