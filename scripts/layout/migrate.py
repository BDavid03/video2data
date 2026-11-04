#!/usr/bin/env python3
"""Move the checkout to the owner's folder layout. Standard library plus git.

  python3 scripts/layout/migrate.py --dry-run | --apply | --verify

The root is the checkout that holds this script, so the same file works in a scratch clone.
One table (MAPPING) drives moves, rewrites, record resolution (src/paths.py reads the same
pairs through scripts/layout/mapping.json) and the environment repair.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "data/layout/journal.json"
# Order matters: `track3` -> `src` first, then `tests` lands inside it.
MAPPING = json.loads(Path(__file__).with_name("mapping.json").read_text())["mapping"]
ENVIRONMENTS = ["linux-cuda", "flashchord/venv", "kit-env", "sam2", "grounding-dino", "hunyuan-mesh",
                "foundationpose", "python", "tools", "uv"]
CODE_SUFFIXES = {".py", ".sh", ".ps1", ".yaml", ".yml", ".toml", ".tex"}
LIVE_DOCS = ["CLAUDE.md", "README.md", "docs/chain/PROGRAM.md", "docs/chain/COMPLETION-PLAN.md",
             "scripts/reproduce/README.md", "scripts/documentation/README.md"]
SKIP_PREFIXES = ("data/recycle/", "data/runs/", "data/reports/", "data/outputs/", "submission/", "upstream/",
                 "docs/chain/handoffs/", "AGENTS.md", "docs/coursework/", "data/templates/", "models/")
NAMES = "|".join(re.escape(old) for old, _ in MAPPING if old not in ("track3", "docs/templates"))
TO = {old: new for old, new in MAPPING}
# a path head: not inside a longer path or word, but allow $ROOT/ and ROOT/ prefixes
HEAD = r"(?:(?<![\w./~-])|(?<=\$ROOT/)|(?<=\$\{ROOT\}/)|(?<=\$PWD/))"
PATH_RE = re.compile(HEAD + r"(" + NAMES + r")(?=/)")
JOIN_RE = re.compile(r"""(\b(?:ROOT|root|project|PROJECT)\s*/\s*)(['"])(""" + NAMES + r""")\2""")
TRACK3_RE = [
    (re.compile(r"\bfrom track3\b"), "from src"),
    (re.compile(r"\bimport track3\b"), "import src"),
    (re.compile(r"(?<![\w./-])track3(?=\.(?:features|paths|provenance|runs|gpu|calibration|metrics|data|timing|tests|__main__)\b)"), "src"),
    (re.compile(r"-m track3\b"), "-m src"),
    (re.compile(r"(?<![\w./-])track3/(?!requests)"), "src/"),
    (re.compile(r"(?<=ROOT / )(['\"])track3\1"), r"\1src\1"),
]
# reviewed one-off literals: (file, old, new)
LITERALS = [
    ("src/provenance.py", '("track3", "configs", "scripts", "tests")', '("src", "data/configs", "scripts")'),
    ("src/__main__.py", '.parts[:2] != ("manifests", "runs")', '.parts[:3] != ("data", "manifests", "runs")'),
    ("scripts/documentation/compile.py", '("docs", "track3", "configs", "scripts", "reports")', '("docs", "src", "data/configs", "scripts", "data/reports")'),
    ("src/features/pose_insurance/scored_scale_pack.py", "('src', 'scripts', 'configs')", "('src', 'scripts', 'data/configs')"),
    ("src/features/policy_training/_common.py", 'ROOT / ".cache" / relative', 'ROOT / "data/.cache" / relative'),
]


def sh(*argv, check=True, cwd=None):
    return subprocess.run(argv, cwd=cwd or ROOT, check=check, text=True, capture_output=True).stdout


def tracked():
    return [n for n in sh("git", "ls-files", "-z").split("\0") if n]


def translate(name):
    for old, new in MAPPING:
        if name == old or name.startswith(old + "/"):
            return new + name[len(old):]
    return name


def rewrite_text(text, name):
    """Boundary-aware rewrite of one file; returns the new text."""
    for pattern, repl in TRACK3_RE:
        text = pattern.sub(repl, text)
    text = PATH_RE.sub(lambda m: TO[m.group(1)], text)
    text = JOIN_RE.sub(lambda m: f'{m.group(1)}{m.group(2)}{TO[m.group(3)]}{m.group(2)}', text)
    text = re.sub(HEAD + r"docs/templates(?=/)", "data/templates", text)
    if name.startswith("src/tests/"):
        text = text.replace("parents[1]", "parents[2]")
    return text


def candidates():
    """(rewrite, skip) lists over the post-move tracked files."""
    rewrite, skipped = [], {}
    for name in sorted({translate(n) for n in tracked()}):
        suffix = Path(name).suffix
        live_doc = name in LIVE_DOCS or (name.endswith("README.md") and name.startswith(("src/", "scripts/", "data/configs/")))
        live_doc = live_doc or bool(re.fullmatch(r"docs/chain/moving-.*-protocol\.md", name))
        if not (suffix in CODE_SUFFIXES or name == ".gitignore" or live_doc):
            if suffix in (".json", ".md", ".txt", ".in", ".log"):
                skipped.setdefault("json/md/txt/log records and non-live docs", 0)
                skipped["json/md/txt/log records and non-live docs"] += 1
            continue
        if name.startswith(SKIP_PREFIXES) and not live_doc:
            skipped.setdefault(f"under {name.split('/')[0]}/ (records, recycle, pinned)", 0)
            skipped[f"under {name.split('/')[0]}/ (records, recycle, pinned)"] += 1
            continue
        if name in ("AGENTS.md", "scripts/layout/migrate.py", "scripts/layout/mapping.json"):
            continue
        rewrite.append(name)
    return rewrite, skipped


def refuse_if_busy():
    if (ROOT / ".runtime/gpu-stage.lock").exists() or (ROOT / "data/.runtime/gpu-stage.lock").exists():
        raise SystemExit("REFUSED: gpu-stage.lock exists")
    scratch = ".runtime/tmp/" in ROOT.as_posix() or "/data/.runtime/tmp/" in ROOT.as_posix()
    if not (scratch or os.environ.get("LAYOUT_SKIP_GPU_CHECK")):
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            raise SystemExit(f"REFUSED: GPU compute apps running: {out!r}")
    me = {os.getpid(), os.getppid()}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in me:
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if not re.search(r"dynamic_feasibility|train_flash_sac|codex exec", cmd) or "migrate.py" in cmd:
            continue
        if scratch and str(ROOT) not in cmd:
            try:
                if not (proc / "cwd").resolve().is_relative_to(ROOT):
                    continue
            except OSError:
                continue
        raise SystemExit(f"REFUSED: live process {proc.name}: {cmd[:160]}")


def save(journal):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    tmp = JOURNAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(journal, indent=1) + "\n")
    os.replace(tmp, JOURNAL)


def is_tracked(rel):
    return bool(sh("git", "ls-files", "--", rel).strip())


def move(old, new, journal, depth=0):
    """git mv for tracked paths (one call per path), plain rename for the untracked remainder."""
    a, b = ROOT / old, ROOT / new
    if not (a.exists() or a.is_symlink()):
        return
    if (b.exists() or b.is_symlink()):
        if a.is_dir() and b.is_dir() and not a.is_symlink() and not b.is_symlink():
            for child in sorted(a.iterdir()):
                move(f"{old}/{child.name}", f"{new}/{child.name}", journal, depth + 1)
            if not any(a.iterdir()):
                a.rmdir()  # an empty directory we just emptied
            return
        raise SystemExit(f"COLLISION: {old} and {new} both exist")
    b.parent.mkdir(parents=True, exist_ok=True)
    if is_tracked(old):
        sh("git", "mv", "--", old, new)
        kind = "git mv"
    else:
        os.rename(a, b)
        kind = "rename"
    journal["moves"].append([old, new, kind])
    save(journal)


def repair_environments(old_roots):
    """Rewrite absolute paths to the old .runtime inside the private environments (temp + replace)."""
    base = ROOT / "data/.runtime"
    new = (ROOT / "data/.runtime").as_posix()
    olds = sorted({f"{r}/.runtime" for r in old_roots}, key=len, reverse=True)
    # a rehearsal clone lives under .runtime/tmp of its parent checkout: never rewrite into it twice
    pats = [re.compile(re.escape(o) + r"(?!/tmp/layout-work)") for o in olds]
    changed = links = 0
    names = re.compile(r"(?:pyvenv\.cfg|direct_url\.json|use-env\.sh|activate.*|.*\.pth|.*\.egg-link|__editable__.*|.*\.cfg|sitecustomize\.py|.*\.pc|.*\.cmake|.*\.la|build\.ninja|.*\.csh|.*\.fish)$")
    for env in ENVIRONMENTS:
        top = base / env
        if not top.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
            for name in [*filenames, *[d for d in dirnames if (Path(dirpath) / d).is_symlink()]]:
                path = Path(dirpath) / name
                if path.is_symlink():
                    target = os.readlink(path)
                    for pat in pats:
                        hit = pat.match(target)
                        if hit:
                            tmp = path.with_name(path.name + ".layout-tmp")
                            os.symlink(new + target[hit.end():], tmp)
                            os.replace(tmp, path)
                            links += 1
                            break
                    continue
                if os.path.basename(dirpath) != "bin" and not names.match(name):
                    continue
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                if len(data) > 2_000_000 or b"\0" in data[:4096]:
                    continue
                text = data.decode("utf-8", "surrogateescape")
                updated = text
                for pat in pats:
                    updated = pat.sub(new, updated)
                if updated != text:
                    tmp = path.with_name(path.name + ".layout-tmp")
                    tmp.write_bytes(updated.encode("utf-8", "surrogateescape"))
                    os.chmod(tmp, path.stat().st_mode)
                    os.replace(tmp, path)
                    changed += 1
    return changed, links


def old_roots():
    roots = {ROOT.as_posix()}
    for cfg in (ROOT / ".runtime/linux-cuda/pyvenv.cfg", ROOT / "data/.runtime/linux-cuda/pyvenv.cfg"):
        if cfg.is_file():
            for line in cfg.read_text().splitlines():
                m = re.match(r"home = (.*?)/(?:data/)?\.runtime/", line)
                if m and not line.split("=")[1].strip().startswith(ROOT.as_posix() + "/data/"):
                    roots.add(m.group(1))
    return roots


def plan():
    rewrite, skipped = candidates()
    return rewrite, skipped


def apply_migration():
    refuse_if_busy()
    journal = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {"moves": [], "rewrites": {}, "state": "new"}
    if journal["state"] == "new":
        journal["head"] = sh("git", "rev-parse", "HEAD").strip()
        journal["roots"] = sorted(old_roots())
        journal["epoch"] = int(sh("git", "log", "-1", "--format=%ct", "HEAD", "--", "docs/report/track3").strip())
    if not (ROOT / "data/layout/inventory-before.json").exists():
        journal["inventory"] = inventory()
    journal["state"] = "moving"
    save(journal)
    started = time.monotonic()
    for old, new in MAPPING:
        move(old, new, journal)
    journal["state"] = "rewriting"
    save(journal)
    rewrite, _ = candidates()
    count = 0
    for name in rewrite:
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        updated = rewrite_text(text, name)
        if updated != text:
            tmp = path.with_name(path.name + ".layout-tmp")  # new inode: hard-linked originals stay intact
            tmp.write_bytes(updated.encode("utf-8", "surrogateescape"))
            os.chmod(tmp, path.stat().st_mode)
            os.replace(tmp, path)
            journal["rewrites"][name] = 1
            count += 1
    for name, old, new in LITERALS:
        path = ROOT / name
        if path.is_file():
            text = path.read_text()
            if old in text:
                tmp = path.with_name(path.name + ".layout-tmp")
                tmp.write_text(text.replace(old, new))
                os.chmod(tmp, path.stat().st_mode)
                os.replace(tmp, path)
    apply_patches(journal)
    journal["state"] = "environments"
    save(journal)
    envs = repair_environments(journal["roots"])
    journal["state"] = "applied"
    journal["seconds"] = round(time.monotonic() - started, 1)
    save(journal)
    print(f"APPLIED: {len(journal['moves'])} moves, {count} files rewritten, environments {envs}, {journal['seconds']} s")


def inventory():
    """Hashes of the packed files and finished run records, by pre-migration name (never edited)."""
    import hashlib
    found = {}
    for pattern in ("submissions/**/*", "outputs/pose_insurance/**/run.json", "outputs/mesh_generative/**/run.json",
                    "outputs/mesh_extents/**/run.json", "outputs/mesh_eval/**/run.json"):
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and not path.is_symlink() and not re.search(r"episode_0{3,4}(41|43|44)\b", path.as_posix()):
                with path.open("rb") as stream:
                    found[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    target = ROOT / "data/layout/inventory-before.json"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(found) + "\n")
    return len(found)


def apply_patches(journal):
    sys.path.insert(0, str(Path(__file__).parent))
    import patches
    done = 0
    for name, old, new in patches.PATCHES:
        new = new.replace("{EPOCH}", str(journal["epoch"]))
        path = ROOT / name
        if old is None:
            if not path.exists():
                path.write_text(new)
                done += 1
            continue
        text = path.read_text()
        if new in text and old not in text:
            continue
        if text.count(old) != 1:
            raise SystemExit(f"PATCH DRIFT: {name} does not contain exactly one copy of the expected text: {old[:70]!r}")
        tmp = path.with_name(path.name + ".layout-tmp")
        tmp.write_text(text.replace(old, new))
        os.chmod(tmp, path.stat().st_mode)
        os.replace(tmp, path)
        done += 1
    return done


def dry_run():
    rewrite, skipped = plan()
    changed = []
    for name in rewrite:
        old_name = next((o for o, n in ((o, n) for o, n in MAPPING) if False), name)
        source = ROOT / name
        # the file may still sit at its pre-move path
        for cand in (name, *[o + name[len(n):] for o, n in MAPPING if name.startswith(n + "/")]):
            if (ROOT / cand).is_file():
                source = ROOT / cand
                break
        if source.is_file():
            text = source.read_text(encoding="utf-8", errors="surrogateescape")
            if rewrite_text(text, name) != text:
                changed.append(name)
    moves = [(o, n) for o, n in MAPPING if (ROOT / o).exists() or (ROOT / o).is_symlink()]
    print(json.dumps({"root": ROOT.as_posix(), "moves": moves, "files_to_rewrite": len(changed),
                      "skipped_classes": skipped, "old_env_roots": sorted(old_roots())}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.apply:
        apply_migration()
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        import checks
        raise SystemExit(checks.run(ROOT, json.loads(JOURNAL.read_text())))


if __name__ == "__main__":
    main()
