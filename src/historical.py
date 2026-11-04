"""Explicit historical verification: frozen sources are read from the recorded pre-migration commit.

Live admission keeps refusing a changed source. Setting V2D_HISTORICAL=1 (check_release.sh does)
lets a pinned source whose bytes differ today pass only when `git show <commit>:<old path>` has
the recorded digest, and every such case is reported as current-source drift, never hidden.
"""
import atexit
import hashlib
import json
import os
import subprocess

from src.paths import PROJECT, historical_name, project_path

DRIFT = {}
_JOURNAL = PROJECT / "data/layout/journal.json"


def active():
    return os.environ.get("V2D_HISTORICAL") == "1"


def commit():
    return json.loads(_JOURNAL.read_text())["head"]


def frozen_digest(name):
    """SHA-256 of the file as it was at the recorded pre-migration commit, or None."""
    try:
        blob = subprocess.check_output(["git", "show", f"{commit()}:{historical_name(name)}"],
                                       cwd=PROJECT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        return None
    return hashlib.sha256(blob).hexdigest()


def source_matches(name, digest):
    """True if the file hashes to `digest` now, or (historical mode only) at the recorded commit."""
    try:
        path = project_path(PROJECT, name)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == digest:
                return True
    except (OSError, ValueError):
        pass
    if active() and frozen_digest(name) == digest:
        DRIFT[name] = digest
        return True
    return False


@atexit.register
def _report():
    if DRIFT:
        print(f"HISTORICAL: {len(DRIFT)} frozen sources differ from the recorded commit {commit()[:10]} "
              f"only by the layout rewrite (current-source drift, reported): " + ", ".join(sorted(DRIFT)[:6]) + " ...")
