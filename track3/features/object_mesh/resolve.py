"""Resolve scan assets unchanged, and only an explicit same-episode spoon reconstruction."""

import shutil
from xml.etree import ElementTree

from track3.provenance import input_file, sha256_file, write_json
from ._common import (FEATURE, ROOT, artifact, authorize, local_path, mesh_metadata,
                      operation_run, producer, read_json)


def _collision(name, snapshot, directory, record):
    """Copy the scan's single unscaled URDF collision mesh; it must share the visual body frame."""
    urdf = local_path(snapshot / f"track_3/public/urdf/{name}_rigid.urdf")
    record["inputs"].append(input_file(ROOT, urdf))
    collisions = ElementTree.parse(urdf).getroot().findall(".//collision")
    if len(collisions) != 1:
        raise ValueError(f"Expected one public collision geometry for {name}")
    node, origin = collisions[0].find("geometry/mesh"), collisions[0].find("origin")
    if (node is None or any(float(value) != 1 for value in node.get("scale", "1 1 1").split()) or
            (origin is not None and any(float(value) != 0 for key in ("xyz", "rpy")
                                       for value in origin.get(key, "0 0 0").split()))):
        raise ValueError("Collision asset must share the visual mesh's unscaled body frame")
    source = local_path(urdf.parent / node.attrib["filename"])
    if not source.is_relative_to(snapshot):
        raise ValueError("Public collision asset escapes the pinned snapshot")
    record["inputs"].append(input_file(ROOT, source))
    destination = directory / f"{name}_collision.glb"
    shutil.copyfile(source, destination)
    return {"collision_file": destination.name, "collision_sha256": sha256_file(destination),
            "collision_source_path": source.relative_to(ROOT).as_posix(),
            "collision_urdf": urdf.relative_to(ROOT).as_posix()}


def resolve(request):
    authorize("resolve", request)
    reconstruction = producer(request.reconstruction_run, "object_mesh.reconstruct", request) if request.reconstruction_run else None
    with operation_run("resolve", request, [reconstruction] if reconstruction else []) as (directory, record):
        import numpy as np
        import trimesh

        source = read_json("configs/sources.lock.json")["dataset"]
        snapshot = local_path(local_path(source["cache_path"]) / "snapshots" / source["revision"])
        rules = read_json(FEATURE / "sources.json")["objects"]
        reconstructed = {}
        if reconstruction:
            path = artifact(reconstruction[0], "mesh.json", record, reconstruction[1]["artifacts"]["mesh.json"])
            reconstructed = {item["name"]: item for item in read_json(path)["objects"]}
        objects, pending = [], []
        for name in record["object_names"]:
            rule = rules.get(name)
            if rule is None:
                raise ValueError(f"No allowed mesh source for {name}")
            if rule["source"] == "public_scan":
                path = local_path(snapshot / rule["path"])
                record["inputs"].append(input_file(ROOT, path))
                item = {"name": name, "source": "public_scan", "body_frame": "scan_frame",
                        "method": "unchanged_asset", "source_path": path.relative_to(ROOT).as_posix()}
            elif rule["source"] == "reconstructed" and name == "wooden_spoon":
                if request.scans_only:
                    pending.append(name)
                    continue
                if name not in reconstructed:
                    raise ValueError("No explicit same-episode wooden_spoon reconstruction")
                item = dict(reconstructed[name])
                if (item["source"] != "reconstructed" or item["method"] != "opening_hull" or
                        item["source_split"] != request.source_split or item["source_episode"] != request.episode_index):
                    raise ValueError("Spoon mesh lineage is not allowed")
                path = artifact(reconstruction[0], item["file"], record, item["sha256"])
            else:
                raise ValueError(f"No permitted automatic mesh fallback for {name}")
            geometry = trimesh.load_mesh(path, process=False)
            if not isinstance(geometry, trimesh.Trimesh):
                raise ValueError("Expected a single triangle mesh asset")
            destination = directory / f"{name}.glb"
            shutil.copyfile(path, destination)
            item.update(file=destination.name, sha256=sha256_file(destination), **mesh_metadata(geometry, np))
            if rule["source"] == "public_scan":
                item.update(_collision(name, snapshot, directory, record))
                record["artifacts"][item["collision_file"]] = item["collision_sha256"]
            objects.append(item)
            record["artifacts"][destination.name] = item["sha256"]
        record["pending_objects"] = pending
        write_json(directory / "mesh.json", {"objects": objects, "pending_objects": pending,
                                             "units": "metres", "provenance": record["provenance"]})
    return directory / "run.json"
