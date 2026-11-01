"""Render completed research records without loading checkpoints or models."""

import csv
from dataclasses import dataclass
import json
import math
import os
import sys
import tempfile

from track3.paths import project_path
from track3.provenance import input_file, read_run
from ._common import ROOT, artifact, authorize, operation_run, read_json


@dataclass(frozen=True)
class ReportRequest:
    investigation_run: str
    opening_result: str | None = None
    external_cuda_run: str | None = None
    synthetic_dataset: str | None = None

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        for name in ("investigation_run", "opening_result", "external_cuda_run", "synthetic_dataset"):
            value = getattr(self, name)
            if (name == "investigation_run" or value is not None) and (
                    not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must name an explicit local artifact")


def _cell(value):
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(rows, columns):
    return ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(_cell(row.get(key)) for key in columns) + " |" for row in rows), ""]


def _score(outcome):
    value = outcome.get("score")
    return value if (outcome.get("status") == "completed" and type(value) in (int, float)
                     and math.isfinite(value) and value >= 0) else None


def _plot(directory, rows, winner):
    """An optional standalone scientific figure; the frozen choice is never recomputed."""
    try:
        cache = artifact(".cache/learning_experiments/matplotlib")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
        sys.pycache_prefix = str(artifact(".cache/learning_experiments/pycache"))
        temporary = artifact(".runtime/learning_experiments/tmp")
        temporary.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(temporary)
        for name in ("TEMP", "TMP", "TMPDIR"):
            os.environ[name] = str(temporary)
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        ranked = sorted((row for row in rows if row["tuning_mae"] is not None),
                        key=lambda row: (row["tuning_mae"], row["candidate"]))
        if not ranked:
            return "No PNG: no admitted tuning scores."
        fig, ax = plt.subplots(figsize=(10, max(4, len(ranked) * 0.45)))
        try:
            ax.scatter([row["tuning_mae"] for row in ranked], [index - 0.12 for index in range(len(ranked))],
                       color="#1675a9", label="Tuning")
            held = [(index, row) for index, row in enumerate(ranked) if row["holdout_mae"] is not None]
            if held:
                ax.scatter([row["holdout_mae"] for _, row in held], [index + 0.12 for index, _ in held],
                           marker="s", color="#b75c18", label="Final holdout")
            ax.set_yticks(range(len(ranked)), [row["candidate"] + ("  [frozen winner]" if row["candidate"] == winner else "")
                                             for row in ranked])
            ax.invert_yaxis()
            ax.set_xlabel("Equal-episode mean absolute natural-log depth error (lower is better)")
            ax.set_title("Tuning order fixed; holdout outcomes do not change selection")
            ax.grid(axis="x", alpha=0.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(directory / "comparison.png", dpi=180, bbox_inches="tight")
        finally:
            plt.close(fig)
        return "![Tuning and holdout comparison](comparison.png)"
    except Exception as error:
        return f"No PNG: {type(error).__name__}: {error}. No dependency installation attempted."


def write_report(request: ReportRequest):
    """Return an owned report.md; all report evidence remains hash-bound inputs."""
    authorize("report")
    inputs, parents, known = {}, set(), {}

    def remember(path):
        path = project_path(ROOT, path)
        entry = input_file(ROOT, path)
        if entry["path"] in known and entry["sha256"] != known[entry["path"]]:
            raise ValueError("Report evidence differs from the investigation input identity")
        if path in inputs and inputs[path] != entry:
            raise ValueError("Report evidence changed while reading")
        inputs[path] = entry
        return path

    def run(path, kind=None):
        path = remember(path)
        value = read_run(ROOT, path, kind=kind)
        if path.name != "run.json":
            raise ValueError("Expected an owning run.json")
        parents.add(path)
        return value

    def derived(entry):
        path = remember(entry["path"])
        if inputs[path] != entry:
            raise ValueError("Derived report input differs from its recorded path/hash")
        owner = run(path.parent / "run.json")
        if owner.get("artifacts", {}).get(path.name) != entry["sha256"]:
            raise ValueError("Report artifact is not owned by its completed producer")
        return read_json(path)

    investigation_path = project_path(ROOT, request.investigation_run)
    investigation = run(investigation_path, "learning_experiments.investigate")
    known.update({entry["path"]: entry["sha256"] for entry in investigation["inputs"]})
    if investigation_path.parent != ROOT / "outputs/learning_experiments/investigate" / investigation["run_id"]:
        raise ValueError("Investigation must remain in its canonical owning directory")
    result = derived({"path": (investigation_path.parent / "result.json").relative_to(ROOT).as_posix(),
                      "sha256": investigation["artifacts"]["result.json"]})
    if project_path(ROOT, investigation["frozen_selection"]["path"]) != investigation_path.parent / "frozen-selection.json":
        raise ValueError("Frozen selection must belong to the investigation being reported")
    frozen = derived(investigation["frozen_selection"])
    winner = frozen["selected_candidate"]
    if (result.get("frozen_selection") != investigation["frozen_selection"]
            or result.get("candidates") != investigation.get("candidates")
            or result.get("selected_candidate") != winner or investigation.get("selected_candidate") != winner
            or frozen.get("selection_group") != "tuning"
            or frozen.get("final_holdout_used_for_selection") is not False
            or result.get("final_holdout_used_for_selection") is not False):
        raise ValueError("Report inputs contradict the frozen tuning selection")
    definitions = {candidate["name"]: candidate for candidate in frozen["candidates"]}
    rows = []
    for candidate in result["candidates"]:
        name = candidate["name"]
        if name not in definitions or any(candidate.get(key) != definitions[name].get(key) for key in
                                          ("technique", "parameters", "checkpoint", "members", "weights")):
            raise ValueError("Candidate definition changed after selection was frozen")
        training = {}
        if candidate.get("checkpoint"):
            checkpoint = project_path(ROOT, candidate["checkpoint"]["path"])
            owner = checkpoint.parent.parent if checkpoint.suffix == ".pt" else checkpoint.parent
            training = run(owner / "run.json")
        elif candidate.get("training", {}).get("failed_run"):
            entry = candidate["training"]["failed_run"]
            path = remember(entry["path"])
            if inputs[path] != entry:
                raise ValueError("Failed training record identity changed")
            training = read_json(path)
        tuning, holdout = candidate.get("tuning", {}), candidate.get("holdout", {})
        rows.append({"candidate": name, "technique": candidate["technique"], "frozen_winner": name == winner,
                     "training_status": candidate.get("training", {}).get("status", "not_applicable"),
                     "training_seconds": training.get("timing", {}).get("wall_seconds"),
                     "peak_allocated_mib": (training["gpu_torch"]["peak_allocated_bytes"] / 2**20
                                            if training.get("gpu_torch", {}).get("peak_allocated_bytes") is not None else None),
                     "device_peak_mib": training.get("gpu", {}).get("peak_sampled_total_device_used_mib"),
                     "tuning_status": tuning.get("status", "unattempted"), "tuning_mae": _score(tuning),
                     "holdout_status": holdout.get("status", "unattempted"), "holdout_mae": _score(holdout),
                     "errors": "\n".join(f"{stage}: {candidate[stage]['error']}" for stage in
                                            ("training", "tuning", "holdout") if candidate.get(stage, {}).get("error"))})
    if len(rows) != len(definitions) or winner not in definitions:
        raise ValueError("Frozen candidate coverage differs from report outcomes")

    baseline_path = project_path(ROOT, investigation["request"]["previous_tuning_run"])
    baseline = run(baseline_path, "learning_experiments.tune")
    baseline_rows = [{"trial": trial["trial_id"], "technique": trial["technique"],
                      "status": trial["status"], "tuning_mae": _score(trial)}
                     for trial in baseline["trials"]]
    baseline_data = {"path": baseline["request"]["dataset"], "sha256": baseline["dataset_sha256"]}
    datasets = {"baseline": derived(baseline_data), **{name: derived(result[name]) for name in
                 ("dataset", "scratch_dataset", "holdout_dataset")}}
    coverage = []
    for name, dataset in datasets.items():
        for episode in sorted({row["episode_index"] for row in dataset["samples"]}):
            samples = [row for row in dataset["samples"] if row["episode_index"] == episode]
            coverage.append({"dataset": name, "episode": episode, "samples": len(samples),
                             "first_frame": min(row["frame_index"] for row in samples),
                             "last_frame": max(row["frame_index"] for row in samples),
                             "first_pts_seconds": min(row["video_pts_seconds"] for row in samples),
                             "last_pts_seconds": max(row["video_pts_seconds"] for row in samples),
                             "valid_target_pixels": sum(row["valid_pixels"] for row in samples)})
    teachers = []
    for episode, entry in result["teacher_runs"].items():
        path = remember(entry["path"])
        if inputs[path] != entry:
            raise ValueError("Teacher producer identity changed")
        teacher = run(path, "depth.generate")
        frames = teacher["frames"]
        teachers.append({"episode": episode, "inferred_frames": teacher["completed_frames"],
                         "frame_stride": teacher["request"].get("frame_stride", 1),
                         "first_frame": min(frame["frame_index"] for frame in frames),
                         "last_frame": max(frame["frame_index"] for frame in frames),
                         "first_pts_seconds": min(frame["video_pts_seconds"] for frame in frames),
                         "last_pts_seconds": max(frame["video_pts_seconds"] for frame in frames)})
    diagnostics = []
    for experiment in result.get("compiler_experiments", []):
        detail = dict(experiment)
        entry = experiment.get("run_identity")
        if entry:
            path = remember(entry["path"])
            if inputs[path] != entry:
                raise ValueError("Compiler outcome identity changed")
            source = read_json(path)
            detail["record"] = {key: source[key] for key in (
                "experiment_status", "backend_executed", "error", "compiler_tools", "eager", "candidate",
                "candidate_total_seconds", "loss_absolute_difference", "output_max_absolute_difference",
                "parameter_gradient_max_absolute_difference", "gpu_torch", "timing") if key in source}
        diagnostics.append(detail)
    optional = {}
    for name, value, kind in (("opening", request.opening_result, "opening_experiments.run"),
                               ("external_cuda", request.external_cuda_run, "scratch_depth.external_cuda_experiment")):
        if value is None:
            continue
        path = remember(value)
        if name == "opening":
            owner = run(path.parent / "run.json", kind)
            optional[name] = derived({**inputs[path], "sha256": owner["artifacts"][path.name]})
        else:
            record = read_json(path)
            if path.name != "run.json" or record.get("kind") != kind or record.get("status") not in ("completed", "failed", "interrupted"):
                raise ValueError("Expected a finished external CUDA experiment record")
            optional[name] = {key: record[key] for key in (
                "kind", "status", "kernel_executed", "error", "cleanup_error", "sample", "torch_version",
                "torch_cuda", "python", "tools", "cuda_home", "model_config", "eager_loss", "cuda_loss",
                "difference_order", "maximum_absolute_differences", "loss_absolute_difference", "tolerance",
                "compile_seconds", "first_eager_seconds", "first_cuda_seconds", "warm_eager_seconds",
                "warm_cuda_seconds", "peak_allocated_bytes", "elapsed_seconds") if key in record}
            if record["status"] == "completed":
                run(path, kind)

    synthetic = None
    if request.synthetic_dataset is not None:
        path = remember(request.synthetic_dataset)
        owner_path = path.parent / "run.json"
        owner = run(owner_path, "synthetic_depth.generate")
        document = derived({**inputs[path], "sha256": owner["artifacts"][path.name]})
        samples = document.get("samples")
        if (path.name != "dataset.json" or path.parent != ROOT / "outputs/synthetic_depth" / owner["run_id"]
                or document.get("kind") != "synthetic_depth_training_pairs"
                or document.get("status") != "completed" or document.get("training_only") is not True
                or document.get("data_origin") != "procedural_synthetic"
                or project_path(ROOT, document["producer_run"]) != owner_path
                or not isinstance(samples, list) or not samples
                or type(owner.get("generated_samples")) is not int or owner["generated_samples"] != len(samples)):
            raise ValueError("Synthetic evidence must retain its separate training-only procedural contract")
        synthetic = {"dataset": inputs[path], "generated_samples": len(samples), **{
            key: document.get(key) for key in ("kind", "data_origin", "training_only", "target_kind", "target_units",
                                              "reference_camera_source", "render_limits")}}

    with operation_run("report", request, inputs=list(inputs), parents=sorted(parents)) as (directory, record):
        record.update(selected_candidate=winner, investigation_run=inputs[investigation_path],
                      frozen_selection=investigation["frozen_selection"], final_holdout_used_for_selection=False,
                      model_execution=False, training_executed=False, prediction_executed=False)
        with (directory / "candidates.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        text = ["# Research campaign report", "", f"Frozen tuning-selected winner: **{winner}**.", "",
                "Holdout outcomes are reported without reselecting the winner. Scores compare to MoGe2 pseudo-depth, "
                "not measured ground truth or official evaluation. Episode 41 is excluded.", "",
                "## Candidate outcomes", "", "[Complete candidate data and exact errors](candidates.csv)", ""]
        text += _table(rows, list(rows[0]))
        text += [_plot(directory, rows, winner), "", "## Dataset and temporal coverage", "",
                 "Baseline data are separate: their early-frame scores do not measure the expanded sparse-clip coverage. "
                 "The holdout dataset also retains training/tuning rows; scoring selects only its declared group.", ""]
        text += _table(coverage, list(coverage[0]))
        text += ["Teacher inference coverage before preparation rejection:", ""]
        text += _table(teachers, list(teachers[0]))
        text += ["## Separate prior baseline", "", f"Prior selection: `{baseline.get('best_trial_id')}`. "
                 "The prior run did not assess final holdout. Its target/frame coverage differs from this campaign; "
                 "these baseline values are not scores on the expanded dataset.", ""]
        text += _table(baseline_rows, ["trial", "technique", "status", "tuning_mae"])
        text += ["## Interpretation", "", "Scratch variants use isolated train-only data and one seed. Training wall times "
                 "include complete owned operations; the reused incumbent's time is its historical training time. "
                 "These are not pure kernel timings or statistically established speedups. Device memory includes other "
                 "applications and sampled telemetry can miss peaks; absent memory is not zero.", "",
                 "The baseline remains a separate 12-trial comparison. Expanded recipes and ensembles were fixed on tuning "
                 "before holdout labels were generated. No holdout-selected winner is computed.", "",
                 "## Compiler outcomes", "", "Exact recorded outcomes follow; a failed attempt is not a passing backend. "
                 "One real batch cannot establish general correctness or steady-state performance.", "",
                 "```json", json.dumps(diagnostics, indent=2), "```", "",
                 "## External CUDA and opening geometry evidence", "",
                 "External CUDA differences, tolerances and timing apply only to the recorded batch/environment. "
                 "Opening stages retain completed/failed/blocked states; successful geometry stages do not establish "
                 "object identity, mesh completeness, camera accuracy or submission readiness. Unprovided optional evidence "
                 "is unavailable, not successful.", "", "```json", json.dumps(optional, indent=2), "```", "",
                 "## Source identities", "", "```json", json.dumps(list(inputs.values()), indent=2), "```", ""]
        if synthetic is not None:
            text += ["## Separate synthetic training data", "",
                     "Only manifest metadata and hashes are read here; no synthetic arrays or training jobs are loaded. "
                     "These procedural samples retain a separate training-only contract and were not mixed into the "
                     "primary campaign's MoGe2 datasets. Generation establishes no measured model improvement. "
                     "Analytic synthetic depths and camera metadata do not establish real-scene fidelity.", "",
                     "```json", json.dumps(synthetic, indent=2), "```", ""]
            record["synthetic_dataset"] = inputs[project_path(ROOT, request.synthetic_dataset)]
        (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
        if any(input_file(ROOT, path) != entry for path, entry in inputs.items()):
            raise ValueError("Report evidence changed during rendering")
        record["report"] = "report.md"
    return directory / "report.md"
