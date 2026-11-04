"""Hash inputs, resume pair trials, and persist kit-score results on CPU."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields, write_json
from track3.runs import ROOT, runtime_flag

FEATURE = Path(__file__).parent
SCRATCH = ".cache/mesh_eval"


@dataclass(frozen=True)
class PairRequest:
    files: dict[str, str]
    roster: str
    output: str
    comparison: tuple[str, str]
    max_trials: int = 52
    time_budget_seconds: int = 7000

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(path.read_text()))

    def __post_init__(self):
        if (len(self.comparison) != 2 or len(set(self.comparison)) != 2
                or not set(self.comparison) <= self.files.keys()):
            raise ValueError("comparison must name two distinct labels from files (before, after)")
        directory = output_path(ROOT, self.output)
        if not directory.is_relative_to(ROOT / "outputs/mesh_eval") or directory == ROOT / "outputs/mesh_eval":
            raise ValueError("Each run needs its own directory under outputs/mesh_eval/")
        if self.max_trials <= 0 or not 0 < self.time_budget_seconds <= 7000:
            raise ValueError("Use a positive trial budget and at most 7000 seconds")


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cpu_runtime() -> dict:
    """Gate evaluation and direct caches to this job's permitted scratch directory."""
    runtime = runtime_flag(FEATURE, "pairs_authorized", "Packed pair evaluation")
    if Path(sys.executable).resolve() != project_path(ROOT, runtime["python"]):
        raise RuntimeError("Run with .runtime/kit-env/bin/python")
    scratch = output_path(ROOT, SCRATCH)
    scratch.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        os.environ[key] = str(scratch)
    tempfile.tempdir = str(scratch)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    return runtime


def prepare(request: PairRequest, request_path: Path, runtime: dict) -> tuple:
    """Read the public roster and scans once and hash all score-producing inputs."""
    from track3.features.mesh_eval import local_cdo

    episodes = json.loads(project_path(ROOT, request.roster).read_text())["episodes"]
    if not set(map(int, episodes)) <= set(runtime["authorized_episode_indices"]) or set(map(int, episodes)) & {41, 43, 44}:
        raise ValueError("Roster contains an unauthorized episode")
    references, scans, pairs = {}, {}, []
    for episode, names in sorted(episodes.items(), key=lambda item: int(item[0])):
        for slot, name in enumerate(names):
            if name not in references and name not in scans:
                try:
                    path = local_cdo.scan_path(name)
                except FileNotFoundError:
                    scans[name] = None
                else:
                    scans[name] = input_file(ROOT, path)
                    references[name] = local_cdo.budgeted(path)
            pairs.append({"episode": int(episode), "slot": slot, "object": name,
                          "seed": local_cdo.CDO._object_seed(int(episode), slot),
                          "status": "covered" if name in references else "no_public_scan"})
    inputs = [request_path, request.roster, FEATURE / "runtime.json",
              *request.files.values(), *[s["path"] for s in scans.values() if s]]
    record = run_fields(ROOT, "mesh_eval", "pairs", asdict(request), inputs=inputs,
                        provenance="public_asset_diagnostic")
    record["sources"] += [input_file(ROOT, p) for p in (
        "track3/features/mesh_eval/local_cdo.py", local_cdo.KIT / "metric_code/track_3/CD-O.py",
        local_cdo.KIT / "v2dlb/mesh_budget.py", local_cdo.KIT / "v2dlb/object_mesh.py")]
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "scipy", "pandas", "pyarrow", "trimesh", "fast-simplification")}
    record.update(pairs=pairs, scans=scans, environment={"python": sys.version, "executable": sys.executable,
                  "packages": versions, "compute": "CPU", "cuda_visible_devices": ""},
                  scorer="unmodified CD-O.py score on one covered pair per trial")
    record["identity"] = identity({"request": asdict(request), "inputs": record["inputs"],
                                   "sources": record["sources"], "packages": versions})
    return record, references, local_cdo.CDO


def load_ledger(path: Path) -> dict:
    """Ignore an interrupted final line while retaining all completed identities."""
    if not path.exists():
        return {}
    rows = {}
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            continue
        if row["status"] == "completed":
            rows[row["identity"]] = row
    return rows


def append_ledger(path: Path, row: dict) -> None:
    """Durably append, replacing only an incomplete tail after an interruption."""
    with path.open("a+b") as stream:
        stream.seek(0)
        data = stream.read()
        if data and not data.endswith(b"\n"):
            tail = data[data.rfind(b"\n") + 1:]
            try:
                json.loads(tail)
            except json.JSONDecodeError:
                stream.truncate(data.rfind(b"\n") + 1)
            else:
                stream.write(b"\n")
        stream.write((json.dumps(row, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def score_pair(label: str, pair: dict, solution, submission, cdo, trial_id: str, run_id: str) -> dict:
    """Call the host entry point and cross-check both slots of episode 1 directly."""
    start = time.perf_counter()
    candidate = submission.loc[submission.row_id.isin(solution.row_id)].copy()
    score = cdo.score(solution.copy(), candidate.copy(), "row_id")
    row = {**pair, "identity": trial_id, "run_identity": run_id, "file": label,
           "status": "completed", "cd_o_cm": score, "completed_at": now()}
    if pair["episode"] == 1:
        merged = solution.merge(candidate, on="row_id", validate="one_to_one")
        direct = cdo._object_chamfer_cm(cdo._to_meshes(merged, cdo._REF_COLUMNS)[pair["slot"]],
                                      cdo._to_meshes(merged, cdo._POINT_COLUMNS)[pair["slot"]], pair["seed"])
        error = abs(score - direct)
        row["equivalence"] = {"score_cm": score, "direct_cm": direct, "absolute_error_cm": error}
        if error > 1e-12:
            raise ValueError(f"score/direct mismatch: {error}")
    row["wall_seconds"] = time.perf_counter() - start
    return row


def write_summary(directory: Path, record: dict, rows: list[dict]) -> None:
    """Publish pair scores, object means, requested label differences and equivalence."""
    from ._metrics import aggregate

    before, after = record["request"]["comparison"]
    labels = sorted(record["request"]["files"])
    summary = aggregate(rows, record["pairs"], (before, after))
    summary.update(run_identity=record["identity"], wall_seconds=record["wall_seconds"],
                   trial_wall_seconds=sum(r["wall_seconds"] for r in rows),
                   equivalence=[{"file": r["file"], "episode": r["episode"], "slot": r["slot"],
                                 **r["equivalence"]} for r in rows if "equivalence" in r],
                   caveat="Public scans, not host references; wooden_spoon has no public scan and is unscored.")
    write_json(directory / "summary.json", summary)
    lines = ["# Exact packed D1 CD-O on public scans", "", summary["caveat"], "",
             f"Coverage: {summary['covered_pairs']}/{summary['total_pairs']} pairs over {len(summary['objects'])} objects; {len(summary['unscored'])} pairs unscored.",
             f"CPU wall time: {record['wall_seconds']:.3f} s; {len(rows)} identity-hashed trials. Lower CD-O is better.", "",
             "| File | Exact pair-weighted mean (cm) | Equal-object mean (cm) |", "| --- | ---: | ---: |"]
    for label, means in sorted(summary["means"].items(), key=lambda item: item[1]["pair_weighted_cm"]):
        lines.append(f"| {label} | {means['pair_weighted_cm']:.8f} | {means['per_object_cm']:.8f} |")
    lines += ["", "## Per pair", "", "Slot order is the kit roster order; each seed is `_object_seed(episode, slot)`.", "",
              "| Episode | Slot | Object | Seed | " + " | ".join(f"{label} (cm)" for label in labels)
              + f" | {after} − {before} (cm) |", "| ---: | ---: | --- | ---: | " + " ---: |" * (len(labels) + 1)]
    for p in summary["pairs"]:
        values = " | ".join(f"{p[label]:.8f}" for label in labels)
        lines.append(f"| {p['episode']} | {p['slot']} | {p['object']} | {p['seed']} | {values} | {p['difference_cm']:+.8f} |")
    lines += ["", "## Per object", "", "| Object | Covered pairs | "
              + " | ".join(f"{label} mean (cm)" for label in labels) + " |",
              "| --- | ---: |" + " ---: |" * len(labels)]
    for o in summary["objects"]:
        values = " | ".join(f"{o[label]:.8f}" for label in labels)
        lines.append(f"| {o['object']} | {o['pairs']} | {values} |")
    lines += ["",
              f"Largest absolute pair difference: {json.dumps(summary['largest_absolute_pair_difference'], sort_keys=True)}.",
              f"Improved/unchanged/worsened pairs: {summary['improved_pairs']}/{summary['unchanged_pairs']}/{summary['worsened_pairs']}.", "",
              "## Host entry-point equivalence", "",
              "Every trial calls unmodified `score(solution, packed_submission, 'row_id')`. On episode 1, both slots for both files also call `_object_chamfer_cm` on the kit-decoded arrays:", ""]
    lines += [f"- {e['file']}, episode {e['episode']}, slot {e['slot']}: score={e['score_cm']:.12f}, direct={e['direct_cm']:.12f}, absolute error={e['absolute_error_cm']:.3g} cm." for e in summary["equivalence"]]
    lines += ["", "The mean of single-pair `score` calls equals its full covered-pair mean because `score` computes the arithmetic mean of independent episode/slot calls. No scorer constants or seeds were changed.", ""]
    (directory / "summary.md").write_text("\n".join(lines))


def evaluate(request: PairRequest, request_path: Path) -> Path:
    """Run the fixed file/pair queue once, reusing only completed matching trials."""
    start = time.perf_counter()
    runtime = cpu_runtime()
    record, references, cdo = prepare(request, request_path, runtime)
    directory = output_path(ROOT, request.output)
    directory.mkdir(parents=True, exist_ok=True)
    run_path, ledger = directory / "run.json", directory / "ledger.jsonl"
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        if previous["identity"] != record["identity"]:
            raise ValueError("Existing run has different inputs/sources; preserve it and use a fresh output entry")
        if previous["status"] == "completed":
            print("Already completed; preserved run and ledger", flush=True)
            return run_path
    record.update(started_at=now(), artifacts={})
    write_json(run_path, record)
    try:
        rows = run_queue(request, record, references, cdo, ledger, start)
        record.update(wall_seconds=time.perf_counter() - start, completed_trials=len(rows))
        write_summary(directory, record, rows)
        record.update(status="completed", completed_at=now(),
                      artifacts={name: input_file(ROOT, directory / name) for name in
                                 ("ledger.jsonl", "summary.json", "summary.md")})
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}",
                      wall_seconds=time.perf_counter() - start)
        raise
    finally:
        write_json(run_path, record)
    return run_path


def run_queue(request: PairRequest, record: dict, references: dict, cdo, ledger: Path, start: float) -> list[dict]:
    """Expand the editable file queue across every covered roster pair."""
    import pandas as pd
    from v2dlb import object_mesh as om

    from ._metrics import solution_frame

    done, rows = load_ledger(ledger), []
    covered = [p for p in record["pairs"] if p["status"] == "covered"]
    if len(covered) * len(request.files) > request.max_trials:
        raise ValueError("Queue exceeds max_trials")
    for label, path in request.files.items():
        submission = pd.read_parquet(project_path(ROOT, path))
        for pair in covered:
            trial_id = identity({"run": record["identity"], "file": label, "pair": pair})
            if trial_id in done:
                rows.append(done[trial_id])
                continue
            if time.perf_counter() - start >= request.time_budget_seconds:
                raise TimeoutError("Time budget reached; rerun the same request to resume")
            solution = solution_frame(pair["episode"], pair["slot"], pair["object"],
                                      references[pair["object"]], om.row_id, om.ARRAYS)
            row = score_pair(label, pair, solution, submission, cdo, trial_id, record["identity"])
            append_ledger(ledger, row)
            rows.append(row)
            print(f"{len(rows)}/{len(covered) * len(request.files)} {label} ep={pair['episode']} slot={pair['slot']} "
                  f"{pair['object']}: {row['cd_o_cm']:.8f} cm ({row['wall_seconds']:.2f}s)", flush=True)
    return rows
