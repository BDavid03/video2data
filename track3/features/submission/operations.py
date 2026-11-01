"""Roster-complete plans, official packers and a separately gated upload ledger."""

import csv
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from track3.provenance import input_file, sha256_file, write_json
from ._campaign import CampaignSelections
from ._packed import admit_packed
from track3.runs import now_utc
from ._common import (LOCK, ROOT, KIT, artifact, artifact_entry, authorize, checked_entry, kit_sources, local_path,
                      operation_run, producer, read_json, submission)

TEMP = ROOT / ".cache/submission-tmp"
POSE_SUFFIXES = ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z",
                 "quaternion_w")


def _rosters(record):
    import pyarrow.parquet as pq
    names = read_json(KIT / "data/track_3_evaluation_objects.json")["episodes"]
    frames = {}
    with (KIT / "data/track_3_sample_submission.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            match = re.fullmatch(r"track_3/e(\d+)/w000/f(\d+)/o(\d+)", row["row_id"])
            if not match:
                raise ValueError("Unexpected Track 3 pose roster key")
            episode, frame, slot = map(int, match.groups())
            frames.setdefault(episode, set()).add((frame, slot))
    for episode, grid in frames.items():
        if episode == 41 or grid != {(t, b) for t in range(max(t for t, _ in grid) + 1) for b in range(len(names[str(episode)]))}:
            raise ValueError("Incomplete or excluded pose roster episode")
    pairs = set()
    for value in pq.ParquetFile(KIT / "data/track_3_mesh_sample_submission.parquet").read(columns=["row_id"]).column(0).to_pylist():
        match = re.fullmatch(r"t3m_(\d{6})_(.+)_(\d)_(\d{6})", value)
        if not match or int(match[1]) == 41 or int(match[3]) not in (0, 1):
            raise ValueError("Unexpected or excluded mesh roster row")
        pairs.add((int(match[1]), match[2]))
    if len(frames) != 20 or len(pairs) != 36:
        raise ValueError("Pinned roster size changed; review the contract")
    return names, frames, pairs


def _git(*arguments):
    return subprocess.run(["git", "-C", str(ROOT), *arguments], capture_output=True, text=True, check=True).stdout.strip()


def _commit():
    """The clean, pushed HEAD of this project's GitHub repository; never creates a repository."""
    if local_path(_git("rev-parse", "--show-toplevel")) != ROOT or _git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Submission requires this project's clean worktree")
    head = _git("rev-parse", "HEAD")
    branch = _git("symbolic-ref", "--quiet", "--short", "HEAD")
    remote = _git("config", "--get", f"branch.{branch}.remote")
    ref = _git("config", "--get", f"branch.{branch}.merge")
    url = _git("remote", "get-url", remote)
    match = re.fullmatch(r"(?:https://github.com/|git@github.com:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?", url)
    if not match or not re.fullmatch("[0-9a-f]{40}", head):
        raise ValueError("Use a credential-free GitHub remote and a full commit")
    pushed = _git("ls-remote", "--exit-code", remote, ref).split()
    if len(pushed) != 2 or pushed != [head, ref]:
        raise ValueError("The clean HEAD is not the remote branch's pushed HEAD")
    return {"code_commit_url": f"https://github.com/{match[1]}/commit/{head}", "commit": head, "remote_ref": ref}


def _candidate(inspection, episode, record):
    report = read_json(artifact(inspection, "inspect.json", record))
    if report.get("blocking") is not False or inspection[1].get("blocking") is not False:
        raise ValueError("Selected rollout did not pass inspection")
    rollout = producer(checked_entry(report["rollout_run"], record), "policy_rollout.rollout", episode)
    if (rollout[1]["checkpoint"] != report["checkpoint"] or rollout[1]["training_run"] != report["training_run"] or
            inspection[1].get("rollout_run") != report["rollout_run"] or
            inspection[1].get("checkpoint") != report["checkpoint"] or
            inspection[1].get("training_run") != report["training_run"] or
            report.get("policy_kind") not in ("resting", "replay", "trained") or
            inspection[1].get("policy_kind") != report["policy_kind"] or rollout[1].get("policy_kind") != report["policy_kind"]):
        raise ValueError("Inspection and rollout lineage differ")
    checkpoint = checked_entry(report["checkpoint"], record)
    training = producer(checked_entry(report["training_run"], record), "policy_training.train", episode)
    if (training[1].get("policy_kind") != report["policy_kind"] or
            training[1].get("reference") != rollout[1]["reference"] or
            report["training_run"] not in rollout[1]["inputs"] or
            report["rollout_run"] not in inspection[1]["inputs"]):
        raise ValueError("Selected rollout differs from its training or inspection inputs")
    owned = [item for item in training[1].get("checkpoints", [])
             if local_path(training[0].parent / item["file"]) == checkpoint and item["sha256"] == report["checkpoint"]["sha256"]]
    if len(owned) != 1:
        raise ValueError("Selected checkpoint is not uniquely owned by this training run")
    if artifact_entry(rollout, f"episode_{episode:06d}.parquet", record) != report["pose_file"]:
        raise ValueError("Inspection did not consume this trajectory")
    return {"inspection_run": input_file(ROOT, inspection[0]), "rollout_run": report["rollout_run"],
            "checkpoint": report["checkpoint"], "pose_file": report["pose_file"],
            "policy_kind": report["policy_kind"], "reference": rollout[1]["reference"]}


def _admit_self_score(parent, candidate, score_parent, scene, episode, record):
    score = read_json(artifact(score_parent, "self_score.json", record))
    if (score["inspection_run"] != input_file(ROOT, parent[0]) or score["scene_run"] != input_file(ROOT, scene[0]) or
            score["rollout_run"] != candidate["rollout_run"] or score_parent[1].get("episode_index") != episode or
            score_parent[1].get("split") != "evaluation" or score.get("policy_kind") != candidate["policy_kind"] or
            score["inspection_run"] not in score_parent[1]["inputs"] or
            score["scene_run"] not in score_parent[1]["inputs"]):
        raise ValueError("Candidate self-score lineage differs")
    if type(score.get("left_working_area")) is not bool:
        raise ValueError("Candidate self-score must state whether it left the working area")
    auc = score["metrics"]["add_auc"]
    if isinstance(auc, bool) or not isinstance(auc, (int, float)) or not math.isfinite(auc):
        raise ValueError("Self-score AUC must be finite")
    candidate.update(self_score=score, self_score_run=input_file(ROOT, score_parent[0]))


def _compare(episode, candidates, resting, scene, scores, pins, request, record):
    """D17: the best in-area challenger replaces the resting fallback only beyond the margin."""
    for parent, candidate in candidates:
        score_parent = scores.get(parent[0])
        if score_parent is None:
            raise ValueError("Every candidate needs a self-score against the pinned tracked scene")
        _admit_self_score(parent, candidate, score_parent, scene, episode, record)
    comparison = {"scene_run": input_file(ROOT, scene[0]),
                  "scene_stale_warnings": pins.stale_warnings(scene),
                  "resting_inspection": input_file(ROOT, resting[0][0]),
                  "selection_margin": request.selection_margin,
                  "candidates": [{"inspection_run": item[1]["inspection_run"],
                                  "self_score_run": item[1]["self_score_run"],
                                  "add_auc": item[1]["self_score"]["metrics"]["add_auc"],
                                  "left_working_area": item[1]["self_score"]["left_working_area"]}
                                 for item in candidates]}
    auc = lambda item: item[1]["self_score"]["metrics"]["add_auc"]
    challengers = [item for item in candidates if item[1]["policy_kind"] in ("replay", "trained")
                   and item[1]["self_score"]["left_working_area"] is False]
    if challengers:
        challenger = max(challengers, key=auc)
        if auc(challenger) > auc(resting) + request.selection_margin:
            return comparison, challenger
    return comparison, None


def _stage_rollout(episode, group, baseline, scores, pins, request, directory, record, names, frames):
    candidates = [(parent, _candidate(parent, episode, record)) for parent in group]
    resting = [item for item in candidates if item[1]["policy_kind"] == "resting"]
    if len(resting) != 1:
        raise ValueError("Name exactly one resting fallback per episode")
    overridden = str(episode) in request.overrides
    selected, route, comparison = resting[0], "resting_default", None
    scene = pins.full_scene(episode) if baseline is not None and not overridden else None
    if scene is not None:
        if read_json(artifact(scene, "scene.json", record))["mode"] != "full":
            raise ValueError("Pinned tracked scene artifact differs from its request")
        comparison, challenger = _compare(episode, candidates, resting[0], scene, scores, pins, request, record)
        if challenger is not None:
            selected, route = challenger, "D17_self_score_margin"
    if overridden:
        matches = [item for item in candidates if item[0][0] == local_path(request.overrides[str(episode)])]
        if len(matches) != 1:
            raise ValueError("Explicit selection must name one of the episode's candidates")
        selected, route = matches[0], "explicit_request"
    parent, candidate = selected
    if baseline is None and candidate["policy_kind"] != "resting":
        raise ValueError("The initial baseline uses resting rollouts for every episode")
    destination = directory / "results" / f"episode_{episode:06d}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # _candidate admitted pose_file's bytes and recorded it as an input.
    shutil.copyfile(local_path(candidate["pose_file"]["path"]), destination)
    return {**candidate, "staged_file": input_file(ROOT, destination), "selection_path": route,
            "stale_warnings": pins.stale_warnings(parent), "selection_evidence": comparison,
            "object_names": names[str(episode)],
            "frame_count": max(t for t, _ in frames[episode]) + 1}


def _stage_mesh(episode, name, parent, pins, directory, record):
    document = read_json(artifact(parent, "mesh.json", record))
    items = [item for item in document["objects"] if item["name"] == name]
    if len(items) != 1:
        raise ValueError("Missing unique reconstructed mesh")
    item = items[0]
    if item["source"] != "reconstructed" or item["source_split"] != "evaluation" or item["source_episode"] != episode:
        raise ValueError("CD-O requires same-episode reconstructed geometry; reuse is not authorized")
    source = artifact_entry(parent, item["file"], record)
    if source["sha256"] != item["sha256"]:
        raise ValueError("Mesh metadata and artifact hashes differ")
    destination = directory / "meshes" / f"episode_{episode:06d}" / (name + ".glb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_path(source["path"]), destination)
    return {"episode_index": episode, "name": name, "run": input_file(ROOT, parent[0]),
            "source_file": source, "staged_file": input_file(ROOT, destination),
            "source": item["source"], "source_episode": episode,
            "stale_warnings": pins.stale_warnings(parent)}


def _completed_plan(path):
    prior = read_json(path)
    return prior.get("kind") == "submission.plan" and prior.get("status") == "completed"


def plan(request):
    authorize("plan")
    if request.baseline_submission and request.single_change == "Initial resting baseline":
        raise ValueError("Describe the single change from the named baseline")
    commit = _commit()
    inspections = {int(key): [producer(value, "policy_rollout.inspect", int(key)) for value in values] for key, values in request.rollouts.items()}
    meshes = {(int(key), name): producer(value, "object_mesh.reconstruct", int(key)) for key, values in request.meshes.items() for name, value in values.items()}
    scores = {local_path(key): producer(value, "policy_rollout.self_score") for key, value in request.self_scores.items()}
    baseline = producer(local_path(request.baseline_submission) / "run.json", "submission.plan") if request.baseline_submission else None
    parents = [item for group in inspections.values() for item in group] + list(meshes.values()) + list(scores.values()) + ([baseline] if baseline else [])
    with operation_run("plan", request, parents) as (directory, record):
        manifest = kit_sources(record)
        lock = read_json(LOCK)
        if baseline is None and [path for path in (ROOT / "submissions").glob("*/run.json")
                                 if path.parent != directory and _completed_plan(path)]:
            raise ValueError("Subsequent submissions must name a completed baseline")
        names, frames, pairs = _rosters(record)
        if set(inspections) != set(frames) or set(meshes) != pairs or any(not group for group in inspections.values()):
            raise ValueError("Plan needs every scored episode and reconstructed mesh pair exactly once")
        if not set(request.overrides) <= {str(episode) for episode in frames}:
            raise ValueError("Unexpected explicit episode selection")
        pins = CampaignSelections(record)
        chosen = {str(episode): _stage_rollout(episode, group, baseline, scores, pins, request, directory, record,
                                               names, frames)
                  for episode, group in inspections.items()}
        mesh_rows = [_stage_mesh(episode, name, parent, pins, directory, record)
                     for (episode, name), parent in sorted(meshes.items())]
        previous = read_json(artifact(baseline, "submission.json", record)) if baseline else None
        changed = [int(key) for key, value in chosen.items() if previous is None or value["rollout_run"] != previous["rollouts"][key]["rollout_run"]]
        previous_meshes = {(row["episode_index"], row["name"]): row["source_file"]["sha256"] for row in previous["meshes"]} if previous else {}
        changed_meshes = [{"episode_index": row["episode_index"], "name": row["name"]} for row in mesh_rows if
                          previous_meshes.get((row["episode_index"], row["name"])) != row["source_file"]["sha256"]]
        write_json(directory / "campaign_selections.json", pins.snapshot())
        document = {**commit, "campaign_selections": input_file(ROOT, directory / "campaign_selections.json"), "rollouts": chosen, "meshes": mesh_rows, "selection_margin": request.selection_margin,
                    "margin_reason": request.margin_reason, "baseline_submission": input_file(ROOT, baseline[0]) if baseline else None,
                    "single_change": request.single_change, "changed_episodes": changed, "changed_meshes": changed_meshes,
                    "kit_manifest": input_file(ROOT, KIT / "KIT_MANIFEST.json"), "kit_evaluator_commit": manifest["evaluator"]["commit"],
                    "kit_archive_sha256": lock["submission_kit"]["archive_sha256"], "flash_chord_commit": lock["policy_runtime"]["commit"],
                    "source_lock": input_file(ROOT, ROOT / "configs/sources.lock.json"), "status": "planned", "provenance": "video_only"}
        write_json(directory / "submission.json", document)
    return directory / "run.json"


def _rollout_checks(check, key, item, parent, names, frames, installation, allowlists, eval_script_sha256, record, np, pq):
    episode = int(key)
    inspection = producer(checked_entry(item["inspection_run"], record), "policy_rollout.inspect", episode)
    actual = _candidate(inspection, episode, record)
    check("inspection_lineage", all(actual[name] == item[name] for name in actual), episode)
    staged = artifact_entry(parent, f"results/episode_{episode:06d}.parquet", record)
    check("unchanged_staged_trajectory", staged == item["staged_file"] and staged["sha256"] == item["pose_file"]["sha256"], episode)
    table = pq.ParquetFile(local_path(staged["path"])).read()
    rows = table.to_pylist()
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    required = {"step", "reference_frame", "time_s", "environment_id", "environment_valid", "object_id", "object_body_id", "object_body_name"}
    required.update("achieved_" + suffix for suffix in POSE_SUFFIXES)
    check("required_columns", required <= set(table.column_names), episode)
    if required <= set(table.column_names):
        grid = [(row["step"], row["environment_id"], row["object_body_id"]) for row in rows]
        check("exact_unique_video_grid", len(grid) == len(frames[episode]) and set(grid) == {(t, 0, b) for t, b in frames[episode]}, episode)
        check("body_names_and_one_body_per_object", all(type(row["object_body_id"]) is int and 0 <= row["object_body_id"] < len(names[key]) and
              row["object_body_name"] == names[key][row["object_body_id"]] and row["object_id"] == row["object_body_id"] for row in rows), episode)
        check("all_worlds_valid", bool(rows) and all(row["environment_valid"] is True for row in rows), episode)
        poses = np.asarray([[row["achieved_" + suffix] for suffix in POSE_SUFFIXES] for row in rows], dtype=float).reshape(-1, 7)
        check("finite_plausible_metres", bool(rows) and np.isfinite(poses).all() and (np.abs(poses[:, :3]) <= 5).all(), episode)
        check("nonzero_quaternions", bool(rows) and (np.linalg.norm(poses[:, 3:], axis=1) >= 1e-8).all(), episode)
        check("reference_clock", all(row["reference_frame"] == row["step"] and isinstance(row["time_s"], (int, float)) and abs(row["time_s"] - row["step"] / 20) < 1e-7 for row in rows), episode)
    expected = {"step_count": str(item["frame_count"]), "control_fps": "20.0", "episode_index": key, "world_count": "1",
                "start_frame": "0", "object_body_count": str(len(names[key])), "position_unit": "meter", "quaternion_order": "xyzw",
                "checkpoint_sha256": item["checkpoint"]["sha256"], "eval_code_sha256": installation["eval_code_sha256"],
                "eval_script_sha256": eval_script_sha256}
    for name, value in expected.items():
        check("metadata." + name, metadata.get("flash_chord." + name) == value, episode)
        if name.endswith("sha256"):
            accepted = allowlists.get("accepted_" + name, [])
            check(name + "_allowlist", not accepted or value in accepted, episode)
    check("reference_lineage", local_path(metadata.get("flash_chord.reference_parquet", "")) == local_path(item["reference"]["path"]), episode)


def _mesh_checks(check, row, parent, record, np, trimesh):
    episode, name = row["episode_index"], row["name"]
    source = producer(checked_entry(row["run"], record), "object_mesh.reconstruct", episode)
    mesh_json = read_json(artifact(source, "mesh.json", record))
    items = [item for item in mesh_json["objects"] if item["name"] == name]
    check("reconstructed_same_episode." + name, len(items) == 1 and items[0]["source"] == "reconstructed" and
          items[0]["source_episode"] == episode and items[0]["source_split"] == "evaluation", episode)
    staged = artifact_entry(parent, f"meshes/episode_{episode:06d}/{name}.glb", record)
    check("unchanged_mesh." + name, staged == row["staged_file"] and staged["sha256"] == row["source_file"]["sha256"], episode)
    checked_entry(row["source_file"], record)
    mesh = trimesh.load(str(local_path(staged["path"])), force="mesh", process=False)
    check("finite_mesh." + name, len(mesh.vertices) > 0 and len(mesh.faces) > 0 and np.isfinite(mesh.vertices).all(), episode)


def _checks(root, parent, record):
    import numpy as np
    import pyarrow.parquet as pq
    import trimesh
    kit_sources(record)
    names, frames, pairs = _rosters(record)
    document = read_json(artifact(parent, "submission.json", record))
    if "campaign_selections" in document:
        if artifact_entry(parent, "campaign_selections.json", record) != document["campaign_selections"]:
            raise ValueError("Submission campaign-selection snapshot changed")
    if set(document["rollouts"]) != {str(episode) for episode in frames}:
        raise ValueError("Submission episode roster differs")
    findings = []

    def check(name, passed, episode=None):
        findings.append({"check": name, "episode_index": episode, "blocking": not bool(passed)})

    installation = read_json("reports/runtime/policy-env.json")
    check("completed_pinned_LF_installation", installation["status"] == "completed" and
          installation["source_commit"] == read_json(LOCK)["policy_runtime"]["commit"])
    allowlists = read_json(KIT / "config/tracks.json")
    eval_script_sha256 = sha256_file(KIT / "flash_chord/scripts/evaluate_policy.py")
    for key, item in document["rollouts"].items():
        _rollout_checks(check, key, item, parent, names, frames, installation, allowlists, eval_script_sha256,
                        record, np, pq)
    check("flat_exact_results", {path.name for path in (root / "results").iterdir()} == {f"episode_{episode:06d}.parquet" for episode in frames})
    check("exact_mesh_pairs", {(row["episode_index"], row["name"]) for row in document["meshes"]} == pairs and len(document["meshes"]) == len(pairs))
    for row in document["meshes"]:
        _mesh_checks(check, row, parent, record, np, trimesh)
    return {"submission_run": input_file(ROOT, parent[0]), "findings": findings, "blocking": any(item["blocking"] for item in findings)}


def check(request):
    authorize("check")
    root, parent = submission(request)
    with operation_run("check", request, [parent], root) as (directory, record):
        report = _checks(root, parent, record)
        write_json(directory / "check.json", report)
        record["blocking"] = report["blocking"]
    return directory / "run.json"


def pack(operation, request):
    if operation not in ("pack-poses", "pack-meshes"):
        raise ValueError("Unknown packing operation")
    authorize(operation)
    root, parent = submission(request)
    inspection = producer(request.check_run, "submission.check")
    with operation_run(operation, request, [parent, inspection], root) as (directory, record):
        report = read_json(artifact(inspection, "check.json", record))
        if report["blocking"] or report["submission_run"] != input_file(ROOT, parent[0]):
            raise ValueError("Require a passing check of this exact submission")
        if _checks(root, parent, record)["blocking"]:
            raise ValueError("Submission inputs no longer pass checks")
        document = read_json(artifact(parent, "submission.json", record))
        if operation == "pack-poses":
            kind, filename = "poses", "track3_poses.csv"
            arguments = [str(KIT / "tools/pack_submission.py"), "--track", "track_3", "--results", str(root / "results"),
                         "--sample", str(KIT / "data/track_3_sample_submission.csv"), "--code-commit", document["code_commit_url"],
                         "--out", str(directory / filename)]
        else:
            kind, filename = "meshes", "track3_meshes.parquet"
            arguments = [str(KIT / "tools/pack_reconstruction.py"), "track3-meshes", "--meshes", str(root / "meshes"),
                         "--sample", str(KIT / "data/track_3_mesh_sample_submission.parquet"), "--commit", document["code_commit_url"],
                         "--out", str(directory / filename)]
        environment = os.environ.copy()
        environment.pop("KAGGLE_API_TOKEN", None)
        environment.update(PYTHONDONTWRITEBYTECODE="1", TMP=str(TEMP), TEMP=str(TEMP))
        TEMP.mkdir(parents=True, exist_ok=True)
        record["command"] = [sys.executable, *arguments]
        with (directory / "packer.log").open("x", encoding="utf-8") as stream:
            subprocess.run(record["command"], cwd=KIT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        admission = admit_packed(kind, directory / filename, parent, record)
        write_json(directory / "packed-check.json", admission)
        record.update(packed_file=admission["file"], packed_admission="passed")
        record["submission_run"] = input_file(ROOT, parent[0])
    return directory / "run.json"


def _reserve(ledger, request, file_entry, metrics, now):
    """Enforce the rolling quota and retry rules before an attempt is recorded."""
    recent = [item for item in ledger["attempts"] if item["kind"] == request.upload_kind and
              datetime.fromisoformat(item["time_utc"]) > now - timedelta(days=7)]
    if len(recent) >= 5:
        raise PermissionError("Rolling seven-day upload quota is exhausted")
    previous = [item for item in ledger["attempts"] if item["file"] == file_entry]
    if previous:
        completed = {metric for item in previous for metric, leg in item["legs"].items() if leg["status"] == "accepted"}
        if not request.only or set(metrics) & completed:
            raise ValueError("Retry only explicit failed or indeterminate legs; accepted legs cannot be resent")
    elif request.only:
        raise ValueError("--only is reserved for retrying a recorded failed upload")


def _send(arguments, environment, token, attempt, ledger, ledger_path, directory):
    """Run the unchanged uploader once and record a redacted outcome per leg."""
    try:
        result = subprocess.run(arguments, cwd=KIT, env=environment, capture_output=True, text=True, check=False)
        output = (result.stdout + "\n" + result.stderr).replace(token, "[REDACTED]")
        for metric, leg in attempt["legs"].items():
            match = re.search(r"^\s*" + re.escape(metric) + r"\s+(OK|FAIL)\s+(.*)$", output, re.MULTILINE)
            leg.update(status=("accepted" if match[1] == "OK" else "failed") if match else "indeterminate",
                       message=match[2] if match else "No confirmed per-leg outcome; inspect Kaggle before retrying")
            quota = re.search(r"^\s*" + re.escape(metric) + r"\s+\S+\s+(\d+) left today", output, re.MULTILINE)
            leg["remaining_daily"] = int(quota[1]) if quota else None
        attempt["exit_code"] = result.returncode
        (directory / "uploader.log").write_text(output, encoding="utf-8")
    except BaseException:
        for leg in attempt["legs"].values():
            leg.update(status="indeterminate", message="Uploader interrupted or could not start")
        raise
    finally:
        attempt["finished_utc"] = now_utc()
        write_json(ledger_path, ledger)
        write_json(directory / "upload-log.json", attempt)


def upload(request):
    authorize("upload")
    root, parent = submission(request)
    packed = producer(request.pack_run, "submission.pack_" + request.upload_kind)
    plan_entry = input_file(ROOT, parent[0])
    if packed[1].get("submission_dir") != root.relative_to(ROOT).as_posix() or packed[1].get("submission_run") != plan_entry:
        raise ValueError("Packed file does not belong to the exact authorized submission directory")
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token or not request.message.strip() or token in request.message:
        raise ValueError("Upload needs an environment token and a non-secret message")
    with operation_run("upload", request, [parent, packed], root) as (directory, record):
        kit_sources(record)
        filename = "track3_poses.csv" if request.upload_kind == "poses" else "track3_meshes.parquet"
        file_entry = artifact_entry(packed, filename, record)
        path = local_path(file_entry["path"])
        admission = read_json(artifact(packed, "packed-check.json", record))
        if (admission.get("schema_version") != 1 or admission.get("kind") != request.upload_kind
                or admission.get("blocking") is not False
                or admission.get("file") != file_entry
                or admission.get("submission_run") != plan_entry
                or packed[1].get("packed_file") != admission["file"]
                or packed[1].get("packed_admission") != "passed"):
            raise ValueError("Upload requires admission of this exact packed file and submission plan")
        track = "track_3" if request.upload_kind == "poses" else "track_3_mesh"
        competitions = read_json(KIT / "config/tracks.json")["tracks"][track]["competitions"]
        if set(request.only) - set(competitions) or len(set(request.only)) != len(request.only):
            raise ValueError("Unknown or duplicate retry legs")
        metrics = request.only or list(competitions)
        ledger_path = ROOT / "submissions/ledger.json"
        lock_path = ROOT / "submissions/ledger.lock"
        owned = False
        try:
            with lock_path.open("x", encoding="utf-8") as stream:
                owned = True
                stream.write(record["run_id"])
            ledger = read_json(ledger_path) if ledger_path.exists() else {"attempts": []}
            now = datetime.now(timezone.utc)
            _reserve(ledger, request, file_entry, metrics, now)
            attempt = {"run_id": record["run_id"], "time_utc": now.isoformat(), "kind": request.upload_kind,
                       "submission_dir": root.relative_to(ROOT).as_posix(), "file": file_entry,
                       "legs": {metric: {"competition": competitions[metric], "status": "pending", "message": None, "remaining_daily": None} for metric in metrics}}
            ledger["attempts"].append(attempt)
            # Reserve quota before the process: a crash or rejection still spends the attempt.
            write_json(ledger_path, ledger)
            arguments = [sys.executable, str(KIT / "submit/v2d_submit.py"), str(path), "--track", track, "--message", request.message]
            if request.only:
                arguments += ["--only", *metrics]
            environment = {key: value for key, value in os.environ.items() if not key.startswith("KAGGLE_")}
            environment.update(KAGGLE_API_TOKEN=token, KAGGLE_CONFIG_DIR=str(directory / "kaggle-config"),
                               PYTHONDONTWRITEBYTECODE="1", TMP=str(TEMP), TEMP=str(TEMP))
            Path(environment["KAGGLE_CONFIG_DIR"]).mkdir(exist_ok=False)
            TEMP.mkdir(parents=True, exist_ok=True)
            _send(arguments, environment, token, attempt, ledger, ledger_path, directory)
            record["upload_status"] = "accepted" if all(leg["status"] == "accepted" for leg in attempt["legs"].values()) else "partial_or_failed"
            if record["upload_status"] != "accepted":
                raise RuntimeError("Upload did not confirm every leg; see the redacted per-leg ledger")
        finally:
            if owned:
                lock_path.unlink()
    return directory / "run.json"
