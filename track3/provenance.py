"""Standard-library provenance, with explicit missing artifacts and no secrets."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


PROVENANCE_TAGS = ("video_only", "public_asset_diagnostic", "public_label_diagnostic")


def input_file(root, path):
    """Hash an explicit local input; a file path alone assigns no provenance."""
    root = Path(root).resolve()
    path = (root / Path(path)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Input resolves outside project root: {path}")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def inherited_provenance(records=(), provenance="video_only", allowed=PROVENANCE_TAGS):
    """Refuse incomplete or untagged producers and inherit the strongest tag."""
    if provenance not in PROVENANCE_TAGS:
        raise ValueError(f"Unknown provenance tag: {provenance!r}")
    tags = [provenance]
    for record in records:
        if record.get("status") != "completed":
            raise ValueError("Only completed input records may be consumed")
        tag = record.get("provenance")
        if tag not in PROVENANCE_TAGS:
            raise ValueError("Untagged legacy inputs and unknown provenance tags are refused")
        tags.append(tag)
    result = max(tags, key=PROVENANCE_TAGS.index)
    if result not in allowed:
        raise ValueError(f"Input provenance {result!r} is not allowed for this operation")
    return result


def read_run(root, path, *, allowed=PROVENANCE_TAGS, kind=None):
    """Read a completed producer; never infer a missing legacy provenance tag."""
    entry = input_file(root, path)
    record = json.loads((Path(root) / entry["path"]).read_text(encoding="utf-8-sig"))
    inherited_provenance([record], allowed=allowed)
    if not isinstance(record.get("kind"), str) or not record["kind"]:
        raise ValueError("Input run has no kind; migrate the producer before consumption")
    if kind is not None and record["kind"] != kind:
        raise ValueError(f"Expected {kind}, got {record['kind']}")
    return record


def run_fields(root, feature, operation, request, *, inputs=(), parents=(),
               provenance="video_only", allowed=PROVENANCE_TAGS):
    """Shared run fields, called only after the owning operation's flag check.

    Raw inputs are named explicitly. Derived inputs also require their producer
    run.json in parents; a consumer must check its artifact's recorded hash.
    """
    root = Path(root).resolve()
    parent_paths = list(parents)
    records = [read_run(root, path, allowed=allowed) for path in parent_paths]
    tag = inherited_provenance(records, provenance, allowed)
    sources = sorted((root / "track3/features" / feature).rglob("*.py"))
    sources += [root / "track3" / name for name in ("paths.py", "gpu.py", "provenance.py", "runs.py")]
    sources.append(root / "configs/sources.lock.json")
    paths = dict.fromkeys(str((root / Path(path)).resolve()) for path in [*inputs, *parent_paths])
    return {"kind": f"{feature}.{operation}", "status": "running", "request": request,
            "inputs": [input_file(root, path) for path in paths],
            "sources": [input_file(root, path) for path in sources], "provenance": tag}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Replace one derived JSON artifact atomically on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def git_evidence(path):
    def run(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    try:
        return {"available": True, "commit": run("rev-parse", "HEAD"),
                "status_porcelain": run("status", "--porcelain"),
                "submodules": run("submodule", "status")}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}


def environment():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "pyarrow",
                 "imageio-ffmpeg", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": packages}


def create_manifest(root, artifacts=(), label="foundation"):
    root = Path(root).resolve()
    lock_path = root / "configs" / "sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    files = set()
    for directory in ("track3", "configs", "scripts", "tests"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"Source file resolves outside project root: {path}")
                files.add(path)
    for name in ("TRACK3-PLAN.md", "README.md", ".gitignore"):
        if (root / name).is_file():
            files.add(root / name)
    for value in artifacts:
        path = Path(value)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Artifact must be inside project root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact missing: {path}")
        files.add(path)
    entries = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                "sha256": sha256_file(path)} for path in sorted(files)]
    source = git_evidence(root / lock["upstream"]["path"])
    source["matches_pin"] = source.get("commit") == lock["upstream"]["commit"]
    source["clean_checkout"] = source.get("available", False) and not source.get("status_porcelain")
    source["source_is_clean_and_pinned"] = source["matches_pin"] and source["clean_checkout"]
    return {"schema_version": 1, "label": label,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": lock, "upstream_checkout": source, "environment": environment(),
            "files": entries,
            "notes": ["Local provenance is not official evaluator validation.",
                      "Dependency presence does not establish GPU/runtime compatibility."]}
