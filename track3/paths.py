"""Constrain generated artifacts to the project, preserving immutable sources."""

import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
_LEGACY_PREFIXES = ("Z:/hugging-face/video2data", "/mnt/z/hugging-face/video2data")


def permitted_roots():
    """Checkout roots allowed to hold generated artifacts (configs/project.json)."""
    config = json.loads((PROJECT / "configs/project.json").read_text(encoding="utf-8-sig"))
    roots = config.get("artifact_roots") or list(_LEGACY_PREFIXES)
    return tuple(str(root).replace("\\", "/").rstrip("/") for root in roots)


def permitted_root():
    """The current checkout, if it is one of the permitted artifact roots."""
    current = PROJECT.resolve()
    text = str(current).replace("\\", "/").casefold()
    for root in permitted_roots():
        if text == root.casefold():
            return current
    raise PermissionError(f"Generated artifacts require a permitted checkout; {current} is not one")


def interpreter(root, value):
    """Resolve a runtime interpreter entry for this platform (string or per-platform object)."""
    if isinstance(value, dict):
        value = value["windows" if os.name == "nt" else "linux"]
    return project_path(root, value)


def project_path(root, value):
    """Read a project path stored in Windows, WSL or local-checkout notation."""
    text = str(value).replace("\\", "/")
    prefixes = (*_LEGACY_PREFIXES, *permitted_roots(), str(Path(root).resolve()).replace("\\", "/"))
    for prefix in prefixes:
        if text.casefold() == prefix.casefold() or text.casefold().startswith(prefix.casefold() + "/"):
            text = text[len(prefix):].lstrip("/")
            break
    path = (Path(root) / text).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes the project")
    return path


def linux_path(root, value):
    """Store a project-owned artifact path for Linux consumers."""
    base = str(Path(root).resolve()).replace("\\", "/")
    if os.name == "nt":
        base = "/mnt/z/hugging-face/video2data"
    return base + "/" + project_path(root, value).relative_to(Path(root).resolve()).as_posix()


def output_path(root, value):
    root = Path(root).resolve()
    canonical = str(root).replace("\\", "/").casefold()
    if canonical not in {r.casefold() for r in permitted_roots()}:
        raise PermissionError("Generated artifacts require a permitted checkout listed in configs/project.json")
    value = Path(value)
    path = (value if value.is_absolute() else root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Generated artifacts must stay inside {root}: {path}")
    relative = path.relative_to(root)
    if not relative.parts or relative.parts[0].startswith("datasets--") or relative.parts[0] in {"upstream", ".git"}:
        raise ValueError(f"Generated artifacts must not overwrite input/source storage: {path}")
    return path
