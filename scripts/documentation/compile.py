"""Combine first-party Markdown without importing or executing project code."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
MARKER = "<!-- video2data:compiled-documentation:v1 -->"
DEFAULT_OUTPUT = "docs/compiled/all-documentation.md"
DOCUMENT_DIRECTORIES = ("docs", "src", "data/configs", "scripts", "data/reports")
SKIP_DIRECTORIES = {
    "compiled", "__pycache__", "node_modules", "upstream", "vendor",
    "datasets", "models", "outputs", "runs", "coursework",
}
PRIORITY = (
    "README.md", "AGENTS.md", "docs/design/legacy-plan.md", "docs/design/README.md",
    "docs/design/system-overview.md", "docs/design/artifact-contracts.md",
    "docs/design/operations.md", "docs/design/version1.md",
    "docs/design/entry-tiers.md", "docs/design/decisions.md",
    "docs/design/error-budget.md", "docs/design/risks-and-open-questions.md",
    "docs/design/roadmap.md", "docs/design/feature-template.md",
)


def generated(path):
    """Recognize our own output without loading an entire previous compilation."""
    with path.open("rb") as stream:
        return stream.read(len(MARKER)).decode("ascii", errors="replace") == MARKER


def documents():
    """Discover only owned documentation trees; never follow symbolic links."""
    paths = [path for path in ROOT.iterdir() if path.suffix.lower() == ".md" and not path.is_symlink() and path.is_file()]
    for name in DOCUMENT_DIRECTORIES:
        directory = ROOT / name
        if directory.is_symlink() or not directory.is_dir():
            continue
        for current, directories, files in os.walk(directory, followlinks=False):
            parent = Path(current)
            directories[:] = sorted(
                name for name in directories
                if not name.startswith(".")
                and name not in SKIP_DIRECTORIES
                and not (parent / name).is_symlink()
            )
            for name in sorted(files):
                path = parent / name
                if (path.suffix.lower() == ".md" and not name.startswith(".")
                        and not path.is_symlink() and path.is_file()):
                    paths.append(path)
    priority = {name: index for index, name in enumerate(PRIORITY)}
    paths.sort(key=lambda path: (priority.get(name := path.relative_to(ROOT).as_posix(), len(priority)), name))
    return [path for path in paths if not generated(path)]


def output_path(value):
    """Constrain output to project storage and refuse symlink destinations."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    # Check the lexical path too: a symlink inside the project must not redirect writes.
    for path in (candidate, *candidate.parents):
        if path == ROOT:
            break
        if path.is_symlink():
            raise ValueError(f"Output may not traverse a symlink: {path}")
    destination = candidate.resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Output must remain inside the project directory")
    relative = destination.relative_to(ROOT)
    if destination.suffix.lower() != ".md":
        raise ValueError("Output must have a .md extension")
    if any(part.startswith(".") for part in relative.parts):
        raise ValueError("Output may not be placed in hidden configuration directories")
    if len(relative.parts) < 3 or relative.parts[:2] != ("docs", "compiled"):
        raise ValueError("Output must be inside docs/compiled/")
    if destination.exists() and (not destination.is_file() or not generated(destination)):
        raise ValueError("Refusing to replace a file not created by this compiler")
    return destination


def source_link(path, destination):
    relative = Path(os.path.relpath(path, destination.parent)).as_posix()
    return quote(relative, safe="/.-_")


def compile_documents(destination):
    paths = documents()
    if not paths:
        raise ValueError("No project Markdown documents were found")
    names = [path.relative_to(ROOT).as_posix() for path in paths]
    entries = [(path, name, "document-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16])
               for path, name in zip(paths, names)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=destination.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(MARKER + "\n\n# Video2Data — collected documentation\n\n")
            stream.write(f"Collected **{len(entries)} Markdown documents** in stable order.\n\n")
            stream.write(
                "This collection includes current documentation and historical reports. "
                "Historical statements retain their original scope and dates. Compilation "
                "does not run or validate the project and grants no execution permission.\n\n"
                "Each section links to its original source and records the SHA-256 of "
                "the bytes read. Document bodies retain their original headings, links "
                "and code blocks; follow the source link to use document-relative links "
                "and heading references in their original context. NUL padding, if "
                "present, is removed only from this collection and reported per source.\n\n"
                "## Contents\n\n"
            )
            for _, name, anchor in entries:
                stream.write(f"- [{name}](#{anchor})\n")
            for path, name, anchor in entries:
                raw = path.read_bytes()
                body = raw.decode("utf-8-sig")
                nul_count = body.count("\x00")
                body = body.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
                digest = hashlib.sha256(raw).hexdigest()
                stream.write(f'\n\n---\n\n<a id="{anchor}"></a>\n\n')
                stream.write(f"## Source: {name}\n\n")
                stream.write(f"[Open original document](<{source_link(path, destination)}>)\n\n")
                stream.write(f"Source SHA-256: `{digest}`\n\n")
                if nul_count:
                    stream.write(f"> Collection note: removed {nul_count:,} NUL characters; source unchanged.\n\n")
                if not body.strip():
                    stream.write("> Collection note: this source contains no readable Markdown text.\n")
                else:
                    stream.write(body)
                    if not body.endswith("\n"):
                        stream.write("\n")
        # Publish only after every source was read, so a failure keeps the previous compilation.
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output .md file inside docs/compiled/; relative paths use the project root",
    )
    args = parser.parse_args()
    try:
        destination = output_path(args.output)
        count = compile_documents(destination)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"Documentation compilation failed: {exc}\n")
    print(f"Wrote {count} documents to {destination}")


if __name__ == "__main__":
    main()
