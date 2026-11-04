"""Project-level commands; stage commands live in their own modules."""

import argparse
from pathlib import Path

from .provenance import create_manifest, write_json
from .paths import output_path


def main():
    parser = argparse.ArgumentParser(description="Track 3 local pipeline tools")
    sub = parser.add_subparsers(dest="command", required=True)
    provenance = sub.add_parser("provenance", help="Record source/configuration/artifact hashes")
    provenance.add_argument("--output", type=Path, required=True)
    provenance.add_argument("--artifact", type=Path, action="append", default=[])
    provenance.add_argument("--artifact-dir", type=Path, action="append", default=[])
    provenance.add_argument("--label", default="foundation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = output_path(root, args.output)
    if output.relative_to(root).parts[:3] != ("data", "manifests", "runs"):
        parser.error("Provenance output must be beneath data/manifests/runs/")
    artifacts = list(args.artifact)
    for value in args.artifact_dir:
        directory = output_path(root, value)
        if not directory.is_dir():
            parser.error(f"Artifact directory does not exist: {directory}")
        artifacts.extend(path for path in directory.rglob("*")
                         if path.is_file() and path.resolve() != output
                         and "__pycache__" not in path.parts and path.suffix != ".pyc")
    if any((path if path.is_absolute() else root / path).resolve() == output for path in artifacts):
        parser.error("Provenance output cannot overwrite a hashed input artifact")
    result = create_manifest(root, artifacts, args.label)
    write_json(output, result)
    print(f"Wrote {output}: {len(result['files'])} hashed files")


if __name__ == "__main__":
    main()
