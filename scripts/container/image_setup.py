#!/usr/bin/env python3
"""Create image-only filesystem aliases for the unchanged frozen tasks and URDFs."""
import json
from pathlib import Path
import sys


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/submission").resolve()
    if root != Path("/submission"):
        raise SystemExit("image_setup is image-only; never create historical aliases on the host")
    for saved, relative in json.loads((root / "path-aliases.json").read_text()).items():
        target = root / relative
        target.resolve().relative_to(root)
        path = Path(saved)
        if not path.is_absolute() or not target.is_file():
            raise ValueError(f"invalid alias: {saved} -> {relative}")
        if path.exists():
            if path.resolve() != target.resolve():
                raise ValueError(f"refusing to replace existing path: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
    print("created frozen task/asset aliases without modifying any artifact")


if __name__ == "__main__":
    main()
