"""Adopt a frozen development winner only after its matched packed veto."""

from dataclasses import asdict, dataclass
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from track3.paths import output_path, project_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json
from track3.runs import ROOT, artifact, new_run_id, recorded, runtime_flag

from ._rule_scores import NAMES, _packed_veto, mesh_for, shape_for
from .build import BuildRequest, FEATURE, ROSTER, build
from .rules import PROTOCOL, read_queue, run


@dataclass(frozen=True)
class AdoptRequest:
    queue: str
    output: str
    incumbent_manifest: str
    frozen_commit: str
    score_cache: str | None = None
    resume_from: str | None = None

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def _frozen_inputs(request: AdoptRequest, queue: dict) -> dict:
    """Admit committed geometry and protocol bytes independently of orchestration edits."""
    paths = [request.queue, queue["attributes"], FEATURE / "_shapes.py", PROTOCOL]
    if queue.get("base_attributes"):
        paths.append(queue["base_attributes"])
    entries = {}
    for value in paths:
        path = project_path(ROOT, value)
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", f"{request.frozen_commit}:{relative}"], cwd=ROOT)
        entry = input_file(ROOT, path)
        if entry["sha256"] != hashlib.sha256(committed).hexdigest():
            raise ValueError(f"Frozen geometry/protocol differs from {request.frozen_commit}: {relative}")
        entries[relative] = entry["sha256"]
    return entries


def _development(request: AdoptRequest, directory: Path, record: dict) -> dict:
    """Run the existing rule loop with an optional copy of content-addressed score caches."""
    output = directory / "development"
    if request.score_cache:
        source = project_path(ROOT, request.score_cache)
        shutil.copytree(source, output / "scores", dirs_exist_ok=True)
        record["inputs"].extend(input_file(ROOT, p) for p in sorted(source.glob("*.json")))
    run_path = run(project_path(ROOT, request.queue), output)
    producer = read_run(ROOT, run_path, kind="mesh_extents.rules")
    summary = artifact((run_path, producer), "summary.json", record)
    return json.loads(summary.read_text())


def _production(build_request: BuildRequest, reuse: str | None, queue: dict, rule: dict, record: dict) -> Path:
    """Admit or build all eleven objects and check reuse against the current frozen constructors."""
    path = project_path(ROOT, reuse) if reuse else build(build_request)
    producer = read_run(ROOT, path, kind="mesh_extents.build", allowed=("video_only",))
    if producer["request"] != asdict(build_request):
        raise ValueError("Production request differs from the selected frozen rule")
    record["inputs"].append(input_file(ROOT, path))
    objects = json.loads(artifact((path, producer), "objects.json", record).read_text())["objects"]
    attributes = json.loads(project_path(ROOT, queue["attributes"]).read_text())["objects"]
    if len(objects) != 11:
        raise ValueError("Production adoption requires all eleven objects")
    for name, row in objects.items():
        shape = shape_for(rule, name, row["extents_m"], build_request.rule_parameter, attributes)
        mesh = mesh_for(shape, rule, name, row["extents_m"], build_request.rule_parameter, attributes)
        if hashlib.sha256(mesh.export(file_type="ply")).hexdigest() != row["sha256"]:
            raise ValueError(f"Production geometry differs from the frozen rule: {name}")
    for name in producer["artifacts"]:
        if name.endswith(".ply"):
            artifact((path, producer), name, record)
    return path



def _check_pack(path: Path, build_run: Path, incumbent: dict, record: dict) -> None:
    """Compare every packed geometry row with the production meshes through the kit budgeter."""
    import numpy as np
    import pandas as pd
    from v2dlb import object_mesh as om
    from v2dlb.mesh_budget import budget_mesh

    command = incumbent["pack_command"]
    sample = project_path(ROOT, command[command.index("--sample") + 1])
    frame = pd.read_parquet(path)
    if not frame["row_id"].equals(pd.read_parquet(sample, columns=["row_id"])["row_id"]):
        raise ValueError("Packed candidate row order differs from the official sample")
    parts = frame["row_id"].str.extract(om.ROW_ID_PATTERN)
    keys = pd.DataFrame({"episode": parts[0].astype(int), "object": parts[1],
                         "array": parts[2].astype(int), "point": parts[3].astype(int)})
    vi, fi = om.ARRAYS.index("object_mesh_vertices"), om.ARRAYS.index("object_mesh_faces")
    for (episode, name), group in keys.groupby(["episode", "object"]):
        mesh = build_run.parent / "meshes" / f"episode_{episode:06d}" / (name + ".ply")
        vertices, faces = budget_mesh(mesh, int((group["array"] == fi).sum()), int((group["array"] == vi).sum()))
        for index, values in ((vi, vertices), (fi, faces)):
            rows = group[group["array"] == index]
            if not np.array_equal(frame.loc[rows.index, ["x", "y", "z"]].to_numpy(), values[rows["point"].to_numpy()]):
                raise ValueError(f"Packed geometry differs from production: {episode}/{name}")
    record["inputs"].append(input_file(ROOT, sample))


def _pack(incumbent: dict, build_run: Path, directory: Path, state: dict, record: dict) -> str:
    """Reuse an admitted pack or invoke the incumbent manifest's official pack command."""
    if state.get("packed_file"):
        packed = project_path(ROOT, state["packed_file"])
        if sha256_file(packed) != state["packed_sha256"]:
            raise ValueError("Packed candidate changed since its recorded veto")
    else:
        packed = directory / "candidate.parquet"
        command = list(incumbent["pack_command"])
        for flag, value in (("--meshes", build_run.parent / "meshes"),
                            ("--commit", incumbent["code_commit_url"]), ("--out", packed)):
            command[command.index(flag) + 1] = str(value)
        with (directory / "pack.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    _check_pack(packed, build_run, incumbent, record)
    entry = input_file(ROOT, packed)
    record["inputs"].append(entry)
    state.update(packed_file=entry["path"], packed_sha256=entry["sha256"])
    return entry["path"]


def _pairs(incumbent: dict, packed: str, directory: Path, state: dict, record: dict) -> dict:
    """Resume the existing pair slice without rewriting completed producer records."""
    from track3.features.mesh_eval.pairs import PairRequest, evaluate

    if state.get("pair_request"):
        path = project_path(ROOT, state["pair_request"])
        request = PairRequest.from_file(path)
        before, after = request.comparison
        if request.files[before] != incumbent["file"] or request.files[after] != packed:
            raise ValueError("Pair request does not compare the admitted incumbent and candidate")
    else:
        path = directory / "pair-request.json"
        request = PairRequest(files={"incumbent": incumbent["file"], "candidate": packed}, roster=ROSTER,
                              output=f"outputs/mesh_eval/pairs/{new_run_id()}", comparison=("incumbent", "candidate"))
        write_json(path, asdict(request))
        state["pair_request"] = path.relative_to(ROOT).as_posix()
    official = json.loads(project_path(ROOT, ROSTER).read_text())["episodes"]
    if json.loads(project_path(ROOT, request.roster).read_text())["episodes"] != official:
        raise ValueError("Packed veto requires the complete official roster in object-slot order")
    record["inputs"].append(input_file(ROOT, ROSTER))
    write_json(directory / "state.json", state)
    pair_run = evaluate(request, path)
    producer = read_run(ROOT, pair_run, kind="mesh_eval.pairs")
    record["inputs"].append(input_file(ROOT, pair_run))
    summary = artifact((pair_run, producer), "summary.json", record,
                       digest=producer["artifacts"]["summary.json"]["sha256"])
    measurements = json.loads(summary.read_text())
    expected = [(int(episode), slot, name)
                for episode, names in sorted(official.items(), key=lambda item: int(item[0]))
                for slot, name in enumerate(names)]
    covered = [pair for pair in expected if pair[2] in NAMES]
    keys = lambda rows: [(r["episode"], r["slot"], r["object"]) for r in rows]
    counts = Counter(name for _, _, name in covered)
    if (len(covered) != 26 or set(counts) != set(NAMES)
            or keys(producer["pairs"]) != expected
            or keys(measurements["pairs"]) != covered
            or measurements["covered_pairs"] != len(covered)
            or measurements["total_pairs"] != len(expected)
            or keys(measurements["unscored"]) != [p for p in expected if p[2] not in NAMES]
            or len(measurements["objects"]) != len(NAMES)
            or {r["object"]: r["pairs"] for r in measurements["objects"]} != dict(counts)):
        raise ValueError("Packed veto requires exactly the official 26 covered pairs over ten objects")
    return {"summary": measurements, "comparison": request.comparison,
            "summary_path": summary.relative_to(ROOT).as_posix()}


def adopt(request: AdoptRequest):
    """Record frozen admission, development selection, production and the single-winner veto."""
    runtime_flag(FEATURE, "evaluation_authorized", "Mesh adoption")
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    queue = read_queue(project_path(ROOT, request.queue))
    frozen = _frozen_inputs(request, queue)
    directory = output_path(ROOT, request.output)
    directory.mkdir(parents=True, exist_ok=True)
    incumbent_path = project_path(ROOT, request.incumbent_manifest)
    incumbent = json.loads(incumbent_path.read_text())
    if sha256_file(project_path(ROOT, incumbent["file"])) != queue["incumbent"]["packed_parquet_sha256"]:
        raise ValueError("Packed incumbent differs from the frozen queue")
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state = {key: state[key] for key in ("winner", "frozen_inputs", "build_run", "packed_file", "packed_sha256", "pair_request") if key in state}
    if state and state["frozen_inputs"] != frozen:
        raise ValueError("Frozen inputs changed; no retuning after selection")
    record = run_fields(ROOT, "mesh_extents", "adopt", asdict(request),
                        inputs=[request.queue, incumbent_path, *frozen], parents=[queue["build"] + "/run.json"],
                        provenance="public_asset_diagnostic")
    run_directory = directory / "runs" / new_run_id()
    with recorded(run_directory, record):
        summary = _development(request, directory, record)
        winner = summary["winner_pending_exact_veto"]
        result = {"winner": winner, "adopt": False, "new_development_scores": summary["new_scores"],
                  "resumed_development_trials": summary["resumed"]}
        if winner is not None:
            if not state and request.resume_from:
                previous = project_path(ROOT, request.resume_from)
                previous_result = json.loads((previous / "veto.json").read_text())
                state = {key: previous_result[key] for key in ("winner", "build_run", "packed_file", "packed_sha256")}
                record["inputs"].append(input_file(ROOT, previous / "veto.json"))
                state["pair_request"] = (previous / "pair-request.json").relative_to(ROOT).as_posix()
            if state.get("winner", winner) != winner:
                raise ValueError("Winner changed; no runner-up substitution")
            state.update(winner=winner, frozen_inputs=frozen)
            write_json(state_path, state)
            rule = next(r for r in queue["rules"] if r["name"] == winner)
            selected = next(r for r in summary["ranking"] if r["rule"] == winner)
            base = read_run(ROOT, ROOT / queue["build"] / "run.json", kind="mesh_extents.build")
            config = {**base["request"], "rule": winner, "rule_parameter": selected["in_sample_parameter"],
                      "rule_queue": request.queue, "attributes_file": queue["attributes"]}
            build_run = _production(BuildRequest(**config), state.get("build_run"), queue, rule, record)
            state["build_run"] = build_run.relative_to(ROOT).as_posix()
            write_json(state_path, state)
            packed = _pack(incumbent, build_run, directory, state, record)
            write_json(state_path, state)
            pairs = _pairs(incumbent, packed, directory, state, record)
            result.update(_packed_veto(pairs["summary"], queue["packed_veto"], pairs["comparison"]),
                          build_run=state["build_run"], packed_file=packed, packed_sha256=state["packed_sha256"],
                          pair_summary=pairs["summary_path"])
        write_json(run_directory / "result.json", result)
        lines = ["# Mesh adoption", "", f"Winner: {winner or 'none'}; adopted: {result['adopt']}.", "",
                 f"New development scores: {summary['new_scores']}; resumed trials: {summary['resumed']}."]
        if winner is not None:
            means = result["candidate_means_cm"]
            lines += ["", f"Packed pair mean: {means['pair_weighted_cm']!r} cm; object mean: {means['per_object_cm']!r} cm.",
                      "", "| Packed veto gate | Verdict |", "| --- | --- |"]
            lines += [f"| {key.replace('_', ' ')} | {'pass' if value else 'fail'} |" for key, value in result["gates"].items()]
            lines += ["", f"Pair evidence: `{result['pair_summary']}`; build: `{result['build_run']}`."]
        lines += ["", "Development evidence on reused public scans; spoon performance remains unmeasured."]
        (run_directory / "summary.md").write_text("\n".join(lines) + "\n")
    return run_directory / "run.json"
