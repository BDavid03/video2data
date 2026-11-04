#!/usr/bin/env python3
"""Measure compressed staging payload without writing an archive or building an image."""
import json
from pathlib import Path
import tarfile
import time

import zstandard

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "data/outputs/container/submission"


class Counter:
    def __init__(self) -> None:
        self.bytes = 0

    def write(self, block: bytes) -> int:
        self.bytes += len(block)
        return len(block)

    def flush(self) -> None:
        pass


def main() -> None:
    start = time.monotonic()
    counter = Counter()
    with zstandard.ZstdCompressor(level=3, threads=0).stream_writer(counter, closefd=False) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            archive.add(STAGE, arcname="submission", recursive=True)
    payload = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file() and not p.is_symlink())
    result = dict(staging_regular_bytes=payload, compressed_staging_tar_bytes=counter.bytes,
                  level=3, elapsed_seconds=time.monotonic() - start,
                  estimated_docker_save_zstd_bytes=[counter.bytes + 150_000_000, counter.bytes + 500_000_000],
                  caveat="Measured staged payload, not a Docker image; base, apt libraries and image-layer overhead are estimated.")
    (STAGE.parent / "size-estimate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
