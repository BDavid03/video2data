"""Constrain generated artifacts to the project, preserving immutable sources."""

import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
_LEGACY_PREFIXES = ("Z:/hugging-face/video2data", "/mnt/z/hugging-face/video2data")


def permitted_roots():
    """Checkout roots allowed to hold generated artifacts (data/configs/project.json)."""
    config = json.loads((PROJECT / "data/configs/project.json").read_text(encoding="utf-8-sig"))
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


def _layout_pairs():
    return [tuple(pair) for pair in json.loads((PROJECT / "scripts/layout/mapping.json").read_text())["mapping"]]


def current_name(value):
    """Name a recorded relative path (old or new layout) in the current layout, by one table."""
    text = str(value).replace("\\", "/")
    for old, new in _layout_pairs():
        if text == old or text.startswith(old + "/"):
            return new + text[len(old):]
    return text


def historical_name(value):
    """The pre-migration name of a current relative path (for `git show <commit>:<name>`)."""
    text = str(value).replace("\\", "/")
    for old, new in reversed(_layout_pairs()):
        if text == new or text.startswith(new + "/"):
            return old + text[len(new):]
    return text


def interpreter(root, value):
    """Resolve a runtime interpreter entry for this platform (string or per-platform object)."""
    if isinstance(value, dict):
        value = value["windows" if os.name == "nt" else "linux"]
    return project_path(root, value)


def project_path(root, value):
    """Read a project path stored in Windows, WSL or local-checkout notation."""
    text = str(value).replace("\\", "/")
    prefixes = (*_LEGACY_PREFIXES, *permitted_roots(), str(Path(root).resolve()).replace("\\", "/"))
    # Longest prefix first: a linked worktree lives inside the main checkout's directory.
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text.casefold() == prefix.casefold() or text.casefold().startswith(prefix.casefold() + "/"):
            text = text[len(prefix):].lstrip("/")
            break
    base = Path(root).resolve()
    if not Path(text).is_absolute():
        text = current_name(text)
    if not Path(os.path.normpath(base / text)).is_relative_to(base):
        raise ValueError("Path escapes the project")
    path = (base / text).resolve()
    # Only links may leave the root: a worktree's .runtime links resolve into the main checkout.
    inside = [base, *(Path(r) for r in permitted_roots() if Path(r).is_absolute())]
    if not any(path.is_relative_to(b) for b in inside):
        raise ValueError("Path escapes the project")
    return path


def linux_path(root, value):
    """Store a project-owned artifact path for Linux consumers."""
    if os.name != "nt":  # the resolved path (possibly through a worktree link) is what Linux reads
        return project_path(root, value).as_posix()
    return "/mnt/z/hugging-face/video2data/" + project_path(root, value).relative_to(Path(root).resolve()).as_posix()


def output_path(root, value):
    root = Path(root).resolve()
    canonical = str(root).replace("\\", "/").casefold()
    if canonical not in {r.casefold() for r in permitted_roots()}:
        raise PermissionError("Generated artifacts require a permitted checkout listed in data/configs/project.json")
    value = Path(value)
    if not value.is_absolute():
        value = Path(current_name(value.as_posix()))
    path = (value if value.is_absolute() else root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Generated artifacts must stay inside {root}: {path}")
    relative = path.relative_to(root)
    if not relative.parts or any(part.startswith("datasets--") for part in relative.parts[:2]) or relative.parts[0] in {"upstream", ".git"}:
        raise ValueError(f"Generated artifacts must not overwrite input/source storage: {path}")
    return path
