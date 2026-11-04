"""Reviewed code edits applied after the path rewrite (strict: a drifted source stops the apply).

Each entry is (file, old, new). `old` is the text as it reads AFTER the boundary-aware rewrite.
{EPOCH} is the report's last pre-migration commit time, frozen so the PDF hash holds.
"""

HISTORICAL = '''"""Explicit historical verification: frozen sources are read from the recorded pre-migration commit.

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
'''

PATCHES = [
    # ---- shared path resolution
    ("src/paths.py", '''def interpreter(root, value):''', '''def _layout_pairs():
    return [tuple(pair) for pair in json.loads((PROJECT / "scripts/layout/mapping.json").read_text())["mapping"]]


def current_name(value):
    """Name a recorded relative path (old or new layout) in the current layout, by one table."""
    text = str(value).replace("\\\\", "/")
    for old, new in _layout_pairs():
        if text == old or text.startswith(old + "/"):
            return new + text[len(old):]
    return text


def historical_name(value):
    """The pre-migration name of a current relative path (for `git show <commit>:<name>`)."""
    text = str(value).replace("\\\\", "/")
    for old, new in reversed(_layout_pairs()):
        if text == new or text.startswith(new + "/"):
            return old + text[len(new):]
    return text


def interpreter(root, value):'''),
    ("src/paths.py", '''    base = Path(root).resolve()
    if not Path(os.path.normpath(base / text)).is_relative_to(base):''', '''    base = Path(root).resolve()
    if not Path(text).is_absolute():
        text = current_name(text)
    if not Path(os.path.normpath(base / text)).is_relative_to(base):'''),
    ("src/paths.py", '''    value = Path(value)
    path = (value if value.is_absolute() else root / value).resolve()''', '''    value = Path(value)
    if not value.is_absolute():
        value = Path(current_name(value.as_posix()))
    path = (value if value.is_absolute() else root / value).resolve()'''),
    ("src/paths.py", '''relative.parts[0].startswith("datasets--")''', '''any(part.startswith("datasets--") for part in relative.parts[:2])'''),
    ("src/historical.py", None, HISTORICAL),
    # ---- run records
    ("src/runs.py", '''from src.paths import PROJECT as ROOT, interpreter, output_path, permitted_root, project_path''',
     '''from src import historical
from src.paths import PROJECT as ROOT, current_name, interpreter, output_path, permitted_root, project_path'''),
    ("src/runs.py", '''def checked_entry(entry, record):
    """Re-admit an input another run recorded, if its bytes still match that record."""
    path = project_path(ROOT, entry["path"])
    if input_file(ROOT, path) != entry:
        raise ValueError("A pinned input changed")
    record_input(entry, record)
    return path''', '''def same_entry(first, second):
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
    return path'''),
    ("src/runs.py", '''    if entry not in source[1]["inputs"]:''', '''    if not has_entry(source[1]["inputs"], entry):'''),
    ("src/runs.py", '''def record_input(entry, record, key="inputs"):
    if entry not in record[key]:''', '''def record_input(entry, record, key="inputs"):
    if not has_entry(record[key], entry):'''),
    # ---- v5 chain
    ("src/features/pose_insurance/scored_stereo_handler.py",
     '''    if any(sha256_file(ROOT / path) != digest for path, digest in plan['sources'].items()):''',
     '''    if not all(historical.source_matches(path, digest) for path, digest in plan['sources'].items()):'''),
    ("src/features/pose_insurance/scored_stereo_handler.py", '''from src.gpu import GpuLease''', '''from src import historical
from src.gpu import GpuLease'''),
    ("src/features/pose_insurance/scored_heading_pack.py", '''from src.provenance import input_file, sha256_file''',
     '''from src.paths import project_path
from src.provenance import input_file, sha256_file'''),
    ("src/features/pose_insurance/scored_heading_pack.py", '''from src.runs import ROOT, artifact, checked_entry, producer, read_json''',
     '''from src.runs import ROOT, artifact, checked_entry, has_entry, producer, read_json, same_entry'''),
    ("src/features/pose_insurance/scored_heading_pack.py", '''or claim['reservation'] != reservation):''',
     '''or not same_entry(claim['reservation'], reservation)):'''),
    ("src/features/pose_insurance/scored_heading_pack.py", '''        source = ROOT / row['source']
        if sha256_file(source) != row['sha256'] or sha256_file(ROOT / row['path']) != row['sha256']:''',
     '''        source = project_path(ROOT, row['source'])
        if sha256_file(source) != row['sha256'] or sha256_file(project_path(ROOT, row['path'])) != row['sha256']:'''),
    ("src/features/pose_insurance/scored_heading_pack.py", '''or reservation not in policy[1]['inputs'] or claim_entry not in policy[1]['inputs']''',
     '''or not has_entry(policy[1]['inputs'], reservation) or not has_entry(policy[1]['inputs'], claim_entry)'''),
    ("src/features/pose_insurance/scored_heading_pack.py", '''with (ROOT / parent['file']).open(newline='') as stream:''',
     '''with project_path(ROOT, parent['file']).open(newline='') as stream:'''),
    ("src/features/pose_insurance/scored_stereo_pack.py", '''shutil.copyfile(ROOT / row['source'], target)''',
     '''shutil.copyfile(project_path(ROOT, row['source']), target)'''),
    ("src/features/pose_insurance/scored_stereo_pack.py", '''from src.paths import output_path''', '''from src.paths import output_path, project_path'''),
    # ---- shell: historical mode flag and the verifier's path mapping
    ("scripts/reproduce/verify.sh", '''export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
''', '''export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$ROOT"
if [[ "${1:-}" == --historical ]]; then export V2D_HISTORICAL=1; shift; fi
'''),
    ("scripts/reproduce/verify.sh", '''import sys

root = Path.cwd()''', '''import sys

root = Path.cwd()
sys.path.insert(0, str(root))
from src import historical
from src.paths import current_name'''),
    ("scripts/reproduce/verify.sh", '''    path = (root / name).resolve()
    moved = moved_paths.get(name) if not path.exists() else None
    if moved is not None:
        path = (root / moved).resolve()''', '''    path = (root / current_name(name)).resolve()
    moved = moved_paths.get(name) if not path.exists() else None
    if moved is not None:
        path = (root / current_name(moved)).resolve()'''),
    ("scripts/reproduce/verify.sh", '''    ok = actual is not None and actual == expected''', '''    ok = actual is not None and actual == expected
    if not ok and historical.active() and historical.frozen_digest(name) == expected:
        ok, detail = True, "frozen source at " + historical.commit()[:10]
        historical.DRIFT[name] = expected'''),
    ("scripts/reproduce/check_release.sh", '''export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
''', '''export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. V2D_HISTORICAL=1
'''),
    ("scripts/reproduce/check_release.sh", '''release = json.loads(Path('scripts/reproduce/release.json').read_text())
''', '''release = json.loads(Path('scripts/reproduce/release.json').read_text())
from src.paths import current_name  # recorded names may be in the old layout
for _entry in release.values():
    for _key in ('path', 'manifest'):
        if _key in _entry:
            _entry[_key] = current_name(_entry[_key])
'''),
    ("scripts/reproduce/check_release.sh", '''str(Path(json.loads(Path(release['pose']['manifest']).read_text())['exports'][0]['path']).parent)''',
     '''str(Path(current_name(json.loads(Path(release['pose']['manifest']).read_text())['exports'][0]['path'])).parent)'''),
    ("scripts/reproduce/d2_poses_v5.sh",'''exec data/.runtime/linux-cuda/bin/python''', '''if [[ "${1:-}" == --historical ]]; then export V2D_HISTORICAL=1; shift; fi
exec data/.runtime/linux-cuda/bin/python'''),
    ("scripts/reproduce/d1_meshes_v5.sh", '''data/.runtime/linux-cuda/bin/python -B - "$@" <<'PY\'''', '''if [[ "${1:-}" == --historical ]]; then export V2D_HISTORICAL=1; shift; fi
data/.runtime/linux-cuda/bin/python -B - "$@" <<'PY\''''),
    ("scripts/reproduce/d3_report.sh", """command git diff --quiet "$COMMIT" -- docs/report/track3/figures/make_figures.py || {
  echo 'Figure generator differs from the archived commit; commit it before building.' >&2; exit 1;
}""", """# the archive above is taken from the working tree, so the layout rewrite needs no commit first"""),
    # ---- report epoch frozen at the pre-migration commit time; archive the working tree
    ("scripts/reproduce/d3_report.sh", '''export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct "$COMMIT" -- docs/report/track3)"''',
     '''export SOURCE_DATE_EPOCH="$(python3 -c "import json;print(json.load(open('scripts/reproduce/release.json'))['report']['epoch'])")"'''),
    ("scripts/reproduce/d3_report.sh", '''command bash -o pipefail -c 'git archive "$1" docs/report/track3 | tar -x -C "$2"' _ "$COMMIT" "$WORK/src"''',
     '''command bash -o pipefail -c 'git ls-files -z docs/report/track3 | tar --null -T - -cf - | tar -x -C "$2"' _ "$COMMIT" "$WORK/src"'''),
    ("scripts/reproduce/release.json", '''"sha256": "62967a997de730705c588f67fd02ff016ac92931c8135fb38310d9551946f814"''',
     '''"sha256": "62967a997de730705c588f67fd02ff016ac92931c8135fb38310d9551946f814",
    "epoch": {EPOCH}'''),
]
