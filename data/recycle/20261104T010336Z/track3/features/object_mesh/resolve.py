"""Resolve each object to an explicit same-episode reconstruction; public scans are refused.

The challenge FAQ requires submitted and simulated meshes to be our own reconstructions, so
`sources.json` lists every object as `reconstructed` and any other source fails closed.
"""

import shutil

from track3.provenance import sha256_file, write_json
from ._common import FEATURE, artifact, authorize, mesh_metadata, operation_run, producer, read_json

ALLOWED_METHODS = ("opening_hull", "opening_tsdf")


def resolve(request):
    authorize("resolve", request)
    reconstruction = producer(request.reconstruction_run, "object_mesh.reconstruct", request) if request.reconstruction_run else None
    with operation_run("resolve", request, [reconstruction] if reconstruction else []) as (directory, record):
        import numpy as np
        import trimesh

        rules = read_json(FEATURE / "sources.json")["objects"]
        reconstructed = {}
        if reconstruction:
            path = artifact(reconstruction[0], "mesh.json", record, reconstruction[1]["artifacts"]["mesh.json"])
            reconstructed = {item["name"]: item for item in read_json(path)["objects"]}
        objects, pending = [], []
        for name in record["object_names"]:
            rule = rules.get(name)
            if rule is None or rule["source"] != "reconstructed":
                raise ValueError(f"No permitted mesh source for {name}: only reconstructions are allowed")
            if request.scans_only:
                pending.append(name)
                continue
            if name not in reconstructed:
                raise ValueError(f"No explicit same-episode reconstruction of {name}")
            item = dict(reconstructed[name])
            if (item["source"] != "reconstructed" or item["method"] not in ALLOWED_METHODS or
                    item["source_split"] != request.source_split or item["source_episode"] != request.episode_index):
                raise ValueError(f"Mesh lineage of {name} is not allowed")
            path = artifact(reconstruction[0], item["file"], record, item["sha256"])
            geometry = trimesh.load_mesh(path, process=False)
            if not isinstance(geometry, trimesh.Trimesh):
                raise ValueError("Expected a single triangle mesh asset")
            destination = directory / f"{name}.glb"
            shutil.copyfile(path, destination)
            item.update(file=destination.name, sha256=sha256_file(destination), **mesh_metadata(geometry, np))
            objects.append(item)
            record["artifacts"][destination.name] = item["sha256"]
        record["pending_objects"] = pending
        write_json(directory / "mesh.json", {"objects": objects, "pending_objects": pending,
                                             "units": "metres", "provenance": record["provenance"]})
    return directory / "run.json"
