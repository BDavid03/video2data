"""Evaluate the two frozen M0 checkpoints and measure them in a separate CPU process."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from track3.gpu import GpuLease
from track3.paths import output_path
from track3.provenance import run_fields, sha256_file, write_json
from track3.runs import heavy_execution, recorded
from . import handler

OUTPUT = handler.OUTPUT / "m0"
PROTOCOL = handler.ROOT / "docs/chain/moving-m0-protocol.md"
CHECKPOINTS = (
    ".runtime/tmp/dynamic-scaling-seed0-k62pzjmq/training/policy_21580800.safetensors",
    ".runtime/tmp/dynamic-scaling-seed1-kzy3z0pr/training/policy_22604800.safetensors",
)
BASELINES = {"Z": "Z-seed0", "L1-seed0": "L1-seed0", "L1-seed1": "L1-seed1"}
HARNESS = handler.FC / "src/flash_chord/evaluation/harness.py"


def runtime(flag):
    """Require M0's independent authorization in the shared slice configuration."""
    settings = json.loads((handler.FEATURE / "runtime.json").read_text())
    settings = {**settings, **settings["M0"]}
    if settings["execution"].get(flag) is not True:
        raise PermissionError(f"M0 requires {flag}")
    return settings


def cpu_measure(phase, env, record):
    """Run M0 measurement with the configured CPU interpreter."""
    settings = runtime("measure_authorized")
    command = [str(handler.ROOT / settings["measurement_python"]), "-B", "-m",
               "track3.features.dynamic_feasibility.partial_eval", "--measure", phase]
    record.setdefault("commands", []).append(command)
    with (OUTPUT / f"{phase}.log").open("w") as stream:
        subprocess.run(command, cwd=handler.ROOT, env={**env, "CUDA_VISIBLE_DEVICES": ""},
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return json.loads((OUTPUT / f"{phase}.json").read_text())


def positions(directory):
    """Read the evaluator's ordered world-frame achieved positions."""
    import numpy as np
    import pyarrow.parquet as pq

    frame = pq.read_table(directory / "trajectory.parquet").to_pandas().sort_values(
        ["reference_frame", "environment_id", "object_body_id"])
    xyz = frame[[f"achieved_position_{axis}" for axis in "xyz"]].to_numpy(dtype=np.float64)
    return frame, xyz


def motion_metrics(achieved, reference):
    """Collect M0 pure motion functions and displacement diagnostics."""
    import numpy as np
    from ._metrics import motion_score, perpendicular_error, transport

    return {"T": transport(achieved, reference), "Q": motion_score(achieved, reference),
            "perpendicular_cm": perpendicular_error(achieved, reference),
            "max_displacement_cm": float(np.linalg.norm(achieved - achieved[0], axis=-1).max() * 100),
            "end_displacement_cm": float(np.linalg.norm(achieved[-1] - achieved[0]) * 100),
            "reference_end_displacement_cm": float(np.linalg.norm(reference[-1] - reference[0]) * 100)}


def measured_export(directory, task, kit, checkpoint_hash=None):
    """Reuse D0 admission and require its completion, frame clock and checkpoint identity."""
    import numpy as np
    from .measure import score_export

    metrics = score_export(directory, task, kit)
    frame, achieved = positions(directory)
    if (not metrics["completed"] or not np.array_equal(frame.step, np.arange(390))
            or not np.allclose(frame.time_s, np.arange(390) / float(task["fps"]), atol=1e-9, rtol=0)):
        raise ValueError("Invalid environment, incomplete evaluation, or changed frame clock/FPS")
    metadata = metrics["evaluator_metadata"]
    for name, expected in (("control_fps", str(float(task["fps"]))), ("start_frame", "0"),
                           ("step_count", "390"), ("world_count", "1"), ("object_body_count", "1")):
        if metadata.get(f"flash_chord.{name}") != expected:
            raise ValueError(f"Invalid export metadata: {name}")
    official = json.loads((directory / "official_metrics.json").read_text())
    if checkpoint_hash and (metadata.get("flash_chord.checkpoint_sha256") != checkpoint_hash
                            or official["checkpoint_sha256"] != checkpoint_hash):
        raise ValueError("Export/checkpoint identity mismatch")
    return {**metrics, **motion_metrics(achieved, task["reference"][:, 0, 0, :3]),
            "reference_end_fraction": official["reference_tracking"]["reference_end_fraction"],
            "success_rate": official["reference_tracking"]["success_rate"],
            "chord_sr": official["metrics"]["chord_sr"],
            "maniptrans_sr": official["metrics"]["maniptrans_sr"], "admission": "VALID"}


def preflight(task, kit):
    """Check the real reference identities, reproduce D0 and score all 390 controls."""
    import numpy as np
    from ._metrics import jump_hold_controls, meets_m0, m0_thresholds, pose_metrics

    reference = task["reference"][:, 0, 0, :3]
    identities = []
    for scale, expected in ((1., (1., 1., 0.)), (0., (0., 0., 0.)), (.5, (.5, .75, 0.))):
        row = motion_metrics(reference[0] + scale * (reference - reference[0]), reference)
        errors = {key: abs(row[key] - value) for key, value in zip(("T", "Q", "perpendicular_cm"), expected)}
        identities.append({"scale": scale, "metrics": row, "absolute_errors": errors,
                           "passed": max(errors.values()) <= 1e-9})
    static = np.repeat(task["reference"][:1], 390, axis=0)
    rows = {"S": {**pose_metrics(static, task["reference"], task["vertices"], kit),
                  **motion_metrics(static[:, 0, 0, :3], reference), "completed": None,
                  "reference_end_fraction": None, "success_rate": None, "admission": "COMPUTED"}}
    reproduction = {}
    static_saved = json.loads((handler.OUTPUT / "comparator_S.json").read_text())
    for name, directory in BASELINES.items():
        rows[name] = measured_export(handler.OUTPUT / "runs" / directory, task, kit)
    for name, row in rows.items():
        saved = (static_saved if name == "S" else json.loads(
            (handler.OUTPUT / "runs" / BASELINES[name] / "metrics.json").read_text()))
        errors = {key: abs(row[key] - saved[key]) for key in ("mean_position_cm", "add_auc")}
        reproduction[name] = {"absolute_errors": errors, "passed": max(errors.values()) <= 1e-9}
    thresholds = m0_thresholds(rows, static_saved["mean_position_cm"])
    controls = []
    for j, achieved in enumerate(jump_hold_controls(reference)):
        row = {"j": j, **motion_metrics(achieved, reference),
               "mean_position_cm": float(np.linalg.norm(achieved - reference, axis=-1).mean() * 100)}
        row.update({f"meets_{name}": meets_m0(row, gate) for name, gate in thresholds.items()})
        controls.append(row)
    control_hits = {name: [row["j"] for row in controls if row[f"meets_{name}"]] for name in thresholds}
    write_json(OUTPUT / "jump_hold_controls.json", controls)
    return {"rows": rows, "thresholds": thresholds, "control_hits": control_hits,
            "best_control_by_Q": max(controls, key=lambda row: row["Q"]),
            "best_control_by_T": max(controls, key=lambda row: row["T"]),
            "best_control_by_E": min(controls, key=lambda row: row["mean_position_cm"]),
            "checks": {"1_identities": {"passed": all(row["passed"] for row in identities), "details": identities},
                       "2_reproduction": {"passed": all(row["passed"] for row in reproduction.values()),
                                          "details": reproduction},
                       "5_controls": {"passed": True, "count": len(controls),
                                      "GO_discriminating": not control_hits["GO"],
                                      "SIGNAL_discriminating": not control_hits["SIGNAL"]}}}


def final_measure(task, kit):
    """Admit both new exports without assigning scores to any invalid seed."""
    from ._metrics import m0_verdict, verdict

    result = json.loads((OUTPUT / "preflight.json").read_text())
    record = json.loads((OUTPUT / "run.json").read_text())
    admissions, identities = {}, {}
    for trial in record["trials"]:
        name = trial["name"]
        try:
            directory = handler.ROOT / trial["export_directory"]
            row = measured_export(directory, task, kit, trial["sha256"])
            result["rows"][name] = row
            admissions[name] = {"passed": True, "directory": trial["export_directory"],
                                "frames": row["frames"], "evaluator_metadata": row["evaluator_metadata"]}
        except Exception as error:
            admissions[name] = {"passed": False, "error": f"{type(error).__name__}: {error}",
                                "attempts": trial["attempts"]}
            result["rows"][name] = {"admission": "INVALID", "details": admissions[name]}
        identities[name] = {"passed": trial.get("identity_passed", False),
                            "sha256": trial["sha256"], "metadata": trial["metadata"]}
    result["checks"].update({"3_admission": {"passed": all(x["passed"] for x in admissions.values()),
                                              "details": admissions},
                              "4_checkpoint_identity": {"passed": all(x["passed"] for x in identities.values()),
                                                        "details": identities}})
    result["verdict"] = m0_verdict(result["rows"], result["checks"], result["control_hits"], result["thresholds"])
    result["d0_rule"] = verdict({f"L1-seed{seed}": {
        "status": "completed" if admissions[f"D1p-seed{seed}"]["passed"] else "failed",
        "metrics": result["rows"][f"D1p-seed{seed}"]} for seed in (0, 1)}, result["rows"]["S"])
    return result


def measurement(phase):
    """Own M0's CPU-only task/kit I/O without changing D0/D1 measurement admission."""
    settings = runtime("measure_authorized")
    heavy_execution({**settings, "python": settings["measurement_python"]})
    import numpy as np

    spec = importlib.util.spec_from_file_location("m0_kit_auc", handler.KIT_METRIC)
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    with np.load(handler.OUTPUT / "task.npz", allow_pickle=False) as data:
        task = dict(data)
    if task["reference"].shape != (390, 1, 1, 7) or float(task["fps"]) != 30.:
        raise ValueError("M0 requires the unchanged 390-frame, 30 Hz reference")
    result = preflight(task, kit) if phase == "preflight" else final_measure(task, kit)
    write_json(output_path(handler.ROOT, OUTPUT / f"{phase}.json"), result)


def evaluate(trial, env, deadline, record):
    """Run the fixed evaluator once, allowing one disclosed retry only without an export."""
    source, checkpoint = (handler.ROOT / trial[key] for key in ("source", "checkpoint"))
    trial["identity_passed"] = sha256_file(source) == sha256_file(checkpoint) == trial["sha256"]
    if not trial["identity_passed"]:
        raise ValueError("Checkpoint changed before evaluation")
    for attempt in range(2):
        if time.time() >= deadline:
            trial["budget_exhausted"] = True
            return
        directory = OUTPUT / "runs" / trial["name"] / f"attempt{attempt}"
        attempt_record = {"kind": "dynamic_feasibility.m0_evaluation", "status": "running",
                          "checkpoint_sha256": trial["sha256"]}
        with recorded(directory, attempt_record):
            command = [str(handler.PYTHON), "-B", "-m", "track3.features.dynamic_feasibility._worker",
                       str(handler.EVALUATOR), f"evaluation.checkpoint={checkpoint}",
                       "evaluation.world_count=1", "evaluation.reset_mode=explicit", "evaluation.start_frame=0",
                       f"evaluation.object_trajectories_output={directory / 'trajectory.parquet'}",
                       f"evaluation.metrics_output={directory / 'official_metrics.json'}",
                       f"hydra.run.dir={directory / 'hydra'}", "logging.mode=disabled"]
            try:
                handler.execute(command, directory / "evaluation.log", env, deadline, attempt_record, "evaluation")
            except RuntimeError as error:
                attempt_record["evaluation_error"] = str(error)
            exported = (directory / "trajectory.parquet").exists()
            attempt_record["export_exists"] = exported
        trial["attempts"].append({"directory": str(directory.relative_to(handler.ROOT)),
                                  "export_exists": exported,
                                  "seconds": attempt_record["evaluation_seconds"],
                                  "error": attempt_record.get("evaluation_error")})
        write_json(OUTPUT / "run.json", record)
        if exported:
            trial["export_directory"] = str(directory.relative_to(handler.ROOT))
            return
        if "evaluation_error" not in attempt_record:
            return


def render(result, record):
    """Write the complete human-readable M0 result with units and validity evidence."""
    def number(value):
        return "—" if value is None else (str(value) if isinstance(value, bool) else f"{value:.9f}")

    lines = [f"# M0: {result['verdict']}", "", "Shipped tissue_box_simple only; no challenge episode or submission.", "",
             "| Arm | T | Q | Perpendicular cm | Mean position cm | P90 position cm | Mean orientation deg | ADD-AUC aligned / raw | Max displacement cm | End displacement cm | Reference end cm | Success rate | End fraction | Complete |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name in ("S", "Z", "L1-seed0", "L1-seed1", "D1p-seed0", "D1p-seed1"):
        row = result["rows"][name]
        if row["admission"] == "INVALID":
            lines.append(f"| {name} | INVALID |")
            continue
        values = [number(row.get(key)) for key in ("T", "Q", "perpendicular_cm", "mean_position_cm",
                  "p90_position_cm", "mean_orientation_deg")]
        values += [f"{number(row['add_auc'])} / {number(row['raw_add_auc'])}"]
        values += [number(row.get(key)) for key in ("max_displacement_cm", "end_displacement_cm",
                   "reference_end_displacement_cm", "success_rate", "reference_end_fraction", "completed")]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += ["", "S is computed, so evaluator success and completion fields do not apply.", "",
              f"Thresholds (full precision): `{json.dumps(result['thresholds'])}`.",
              f"D0 position/completion rule: `{json.dumps(result['d0_rule'])}`.", "", "## Validity", ""]
    for name, check in sorted(result["checks"].items()):
        lines.append(f"- {name}: {'PASS' if check['passed'] else 'FAIL'}; details in final.json.")
    lines += ["", "## Jump-and-hold controls", "",
              "Best means greatest Q; maxima of T and minimum E are separately retained in preflight.json.",
              f"Best by Q: `{json.dumps(result['best_control_by_Q'])}`.",
              f"Controls meeting thresholds: `{json.dumps(result['control_hits'])}`.", "",
              "## Checkpoints", "", "| Arm | SHA-256 | Steps | Training voc_scale |", "| --- | --- | ---: | ---: |"]
    for trial in record["trials"]:
        lines.append(f"| {trial['name']} | {trial['sha256']} | {trial['metadata'].get('environment_steps')} | {trial['metadata']['voc_scale']} |")
    lines += ["", f"GPU lease wall time: {record['gpu_seconds']:.6f} seconds (300-second shared cap).",
              f"Peak sampled total device memory: {record['gpu']['peak_sampled_total_device_used_mib']} MiB; may miss brief peaks.",
              f"Retries: {sum(max(0, len(trial['attempts']) - 1) for trial in record['trials'])}.", "",
              "Assistance removal: hashed evaluation/harness.py:230 sets voc_scale=0; :232-233 zero force/torque;",
              ":265-272 zero curriculum VOC; :438 applies it; :450 resets explicitly. Checkpoint training VOC is not evaluation VOC.",
              "", "Commands and hashed inputs/sources: run.json and runs/*/attempt*/run.json.",
              "Every control: jump_hold_controls.json. Pre-evaluation evidence: preflight.json. Admission and verdict: final.json.",
              "", "NULL means this screen was not met; M0 alone cannot kill W3. GO/SIGNAL require the lead's independent recomputation."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")


def run():
    """Record inputs, preflight, evaluate seed0 then seed1 under one lease, and report."""
    runtime("run_authorized")
    env = handler.environment()
    if OUTPUT.exists():
        if (OUTPUT / "finished.json").exists():
            print((OUTPUT / "summary.md").read_text())
            return
        raise FileExistsError("M0 already started; inspect run.json and exports before resuming; never repeat an export")
    from safetensors import safe_open

    inputs = [PROTOCOL, handler.FEATURE / "runtime.json", handler.OUTPUT / "task.npz",
              handler.OUTPUT / "comparator_S.json", handler.EVALUATOR, handler.KIT_METRIC,
              *(handler.ROOT / path for path in CHECKPOINTS)]
    for directory in BASELINES.values():
        inputs.extend(handler.OUTPUT / "runs" / directory / name
                      for name in ("trajectory.parquet", "metrics.json", "official_metrics.json"))
    record = run_fields(handler.ROOT, "dynamic_feasibility", "m0_partial_evaluation",
                        {"protocol_commit": "1ec1e90", "maximum_gpu_seconds": 300,
                         "checkpoints": list(CHECKPOINTS)}, inputs=inputs, provenance="public_asset_diagnostic")
    record.update(pid=os.getpid(), diagnostic_only=True, challenge_episode_access=False, trials=[],
                  task_inputs=[handler.entry(HARNESS)],
                  assistance_removal={"source": handler.entry(HARNESS), "composition_lines": [230, 232, 233],
                                      "curriculum_lines": [265, 271, 272, 438], "explicit_reset_line": 450})
    with recorded(output_path(handler.ROOT, OUTPUT), record):
        for seed, path in enumerate(CHECKPOINTS):
            source = handler.ROOT / path
            with safe_open(str(source), framework="numpy") as policy:
                metadata = policy.metadata()
            if metadata.get("format") != "learner_state" or metadata.get("version") != "1":
                raise ValueError("Unexpected checkpoint format/version")
            metadata = {key.removeprefix("user."): value for key, value in metadata.items() if key.startswith("user.")}
            trial = {"name": f"D1p-seed{seed}", "source": path, "sha256": sha256_file(source),
                     "metadata": metadata, "attempts": []}
            record["trials"].append(trial)
        write_json(OUTPUT / "run.json", record)
        for trial in record["trials"]:
            target = OUTPUT / "checkpoints" / f"{trial['name']}.safetensors"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(handler.ROOT / trial["source"], target)
            trial["checkpoint"] = str(target.relative_to(handler.ROOT))
        write_json(OUTPUT / "run.json", record)
        pre = cpu_measure("preflight", env, record)
        if not all(check["passed"] for check in pre["checks"].values()):
            raise ValueError("M0 pre-evaluation validity checks failed; no policy evaluated")
        main_lock = Path("/home/ceolwolf/repos/video2data/.runtime/gpu-stage.lock")
        if main_lock.exists():
            raise RuntimeError("Main-checkout GPU lease already exists")
        with GpuLease(handler.ROOT, 1024, 4096, stage="dynamic_feasibility_m0") as lease:
            started = time.monotonic()
            deadline = time.time() + 300
            record["gpu_started_epoch"] = time.time()
            try:
                for trial in record["trials"]:
                    evaluate(trial, env, deadline, record)
                    lease.require_headroom()
            finally:
                record.update(gpu_seconds=time.monotonic() - started, gpu=lease.report())
        write_json(OUTPUT / "run.json", record)
        result = cpu_measure("final", env, record)
        render(result, record)
        record["verdict"] = result["verdict"]
        write_json(OUTPUT / "finished.json", {"verdict": result["verdict"], "gpu_seconds": record["gpu_seconds"]})
    print((OUTPUT / "summary.md").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=("preflight", "final"))
    args = parser.parse_args()
    measurement(args.measure) if args.measure else run()


if __name__ == "__main__":
    main()
