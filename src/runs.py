"""Run-record lifecycle, episode authorization and producer admission shared by feature slices."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4

from src import historical
from src.paths import PROJECT as ROOT, current_name, interpreter, output_path, permitted_root, project_path
from src.provenance import input_file, read_run, sha256_file, write_json

SPLIT = "data/configs/splits/public-v1.json"
LOCK = "data/configs/sources.lock.json"
CACHE_FOLDERS = {"TEMP": "data/.runtime/tmp", "TMP": "data/.runtime/tmp", "TMPDIR": "data/.runtime/tmp",
                 "TORCH_HOME": "data/.cache/torch", "TORCH_EXTENSIONS_DIR": "data/.cache/torch-extensions",
                 "TORCHINDUCTOR_CACHE_DIR": "data/.cache/torch-inductor", "TRITON_CACHE_DIR": "data/.cache/triton",
                 "CUDA_CACHE_PATH": "data/.cache/cuda", "HF_HOME": "data/.cache/huggingface", "XDG_CACHE_HOME": "data/.cache/xdg",
                 "MPLCONFIGDIR": "data/.cache/matplotlib", "PYTHONPYCACHEPREFIX": "data/.cache/pycache"}


def read_json(value):
    return json.loads(project_path(ROOT, value).read_text(encoding="utf-8-sig"))


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def new_run_id(label=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return "-".join(part for part in (stamp, label, uuid4().hex[:8]) if part)


def dataset_revision():
    return read_json(LOCK)["dataset"]["revision"]


def runtime_flag(feature_dir, flag, what):
    """Read a slice's runtime.json and refuse unless the execution flag is exactly true."""
    runtime = read_json(feature_dir / "runtime.json")
    if runtime["execution"].get(flag) is not True:
        raise PermissionError(f"{what} remains disabled")
    return runtime


def heavy_execution(runtime):
    """Require the approved interpreter and move every cache/temp location into the checkout before heavy imports."""
    permitted_root()
    if Path(sys.executable).resolve() != interpreter(ROOT, runtime["python"]):
        raise RuntimeError("Run with the project interpreter named in the slice's runtime.json")
    for key, folder in CACHE_FOLDERS.items():
        path = output_path(ROOT, folder)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    tempfile.tempdir = os.environ["TMPDIR"]
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return runtime


def source_hashes(feature_dir, extra=(), suffixes=(".py",)):
    """Hash the slice's sources, the shared primitives, the lock and the split: the code identity of a run."""
    sources = sorted(path for path in feature_dir.rglob("*") if path.is_file() and path.suffix in suffixes)
    sources += [ROOT / "src" / name for name in ("gpu.py", "paths.py", "provenance.py", "runs.py")]
    sources += [ROOT / LOCK, ROOT / SPLIT, *(project_path(ROOT, item) for item in extra)]
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sources}


def save_npz(path, np, **arrays):
    """Write a compressed NPZ atomically next to its final name."""
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def authorize_episode(runtime, request, what):
    """Frozen-split and per-episode allow-list check shared by episode-scoped slices."""
    execution, split = runtime["execution"], read_json(SPLIT)
    if request.source_split == "evaluation":
        if execution.get("evaluation_authorized") is not True:
            raise PermissionError(f"Evaluation {what} remains disabled")
    elif request.episode_index in split["validation_episode_indices"]:
        if execution.get("validation_authorized") is not True:
            raise PermissionError(f"Validation {what} remains disabled")
    elif request.episode_index not in split["development_episode_indices"]:
        raise ValueError("Episode is outside the frozen public split")
    if request.episode_index not in runtime["authorized_episode_indices"].get(request.source_split, []):
        raise PermissionError(f"{what} requires authorization for the named split and episode")


def record_input(entry, record, key="inputs"):
    if not has_entry(record[key], entry):
        record[key].append(entry)
    return entry


def producer(value, kind, request=None, allowed=("video_only",)):
    """Admit one completed producer run.json; an episode request must match its split and episode."""
    if value is None:
        raise ValueError(f"Supply an explicit completed {kind} run.json")
    path = project_path(ROOT, value)
    record = read_run(ROOT, path, kind=kind, allowed=allowed)
    if request is not None and (record.get("split") != request.source_split
                                or record.get("episode_index") != request.episode_index):
        raise ValueError(f"{kind} input belongs to a different split or episode")
    return path, record


def artifact(parent, name, record, digest=None):
    """Admit a producer-owned artifact against its recorded hash and list it as an input.

    `parent` is the producer's run.json path or a (path, record) pair; without an explicit
    digest the producer record's artifact inventory is the reference.
    """
    producer_path, producer_record = parent if isinstance(parent, tuple) else (parent, None)
    path = project_path(ROOT, producer_path.parent / name)
    if not path.is_relative_to(producer_path.parent):
        raise ValueError("Artifact escapes its producer directory")
    if digest is None:
        digest = producer_record["artifacts"][name]
    entry = input_file(ROOT, path)
    if entry["sha256"] != digest:
        raise ValueError(f"Input artifact differs from its producer: {path}")
    record_input(entry, record)
    return path


def same_entry(first, second):
    """Resolved identity plus digest: an old-layout record equals its relocated file."""
    return current_name(first["path"]) == current_name(second["path"]) and first["sha256"] == second["sha256"]


def has_entry(entries, entry):
    return any(same_entry(item, entry) for item in entries)


def checked_entry(entry, record):
    """Re-admit an input another run recorded, if its bytes still match that record."""
    path = project_path(ROOT, entry["path"])
    current = input_file(ROOT, path)
    if not same_entry(current, entry) and not historical.source_matches(entry["path"], entry["sha256"]):
        raise ValueError("A pinned input changed")
    record_input(current, record)
    return path


def require_input(source, parent):
    """The consumer run `source` must list the producer run `parent` among its inputs."""
    entry = input_file(ROOT, parent[0])
    if not has_entry(source[1]["inputs"], entry):
        raise ValueError(f"{source[1]['kind']} did not consume the selected {parent[1]['kind']} run")
    return entry


def same_revision(parents, revision, skip=()):
    if any(record.get("dataset_revision") != revision for _, record in parents if record.get("kind") not in skip):
        raise ValueError("Input dataset revisions differ")
    return revision


@contextmanager
def recorded(directory, record, *, hash_artifacts=True):
    """Create the run directory, persist run.json before and after the body, close frames and artifacts."""
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record.setdefault("started_utc", now_utc())
    write_json(directory / "run.json", record)
    try:
        yield directory, record
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        for row in [*record.get("frames", ()), *record.get("trials", ())]:
            if row.get("status") == "running":
                row.update(status=record["status"], error=record["error"])
        if not hasattr(error, "feature_run_path"):
            try:
                error.feature_run_path = (directory / "run.json").relative_to(ROOT).as_posix()
            except AttributeError:
                pass
        raise
    finally:
        record["finished_utc"] = now_utc()
        record["timing"] = {"wall_seconds": time.perf_counter() - started}
        if hash_artifacts:
            record.setdefault("artifacts", {}).update(
                {path.relative_to(directory).as_posix(): sha256_file(path)
                 for path in directory.rglob("*") if path.is_file() and path.name != "run.json"})
        write_json(directory / "run.json", record)
