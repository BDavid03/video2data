#!/usr/bin/env bash
# Run as the owner with Docker + NVIDIA Container Toolkit access. No uploads or pushes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
: "${DATASET_ROOT:?Set DATASET_ROOT to the released challenge dataset root to mount at /data}"
IMAGE="${IMAGE:-video2data-submission:final}"
BASE_IMAGE="${BASE_IMAGE:-ubuntu:26.04}"
BUILD="$ROOT/data/outputs/container"
PYTHON="${STAGE_PYTHON:-$ROOT/data/.runtime/linux-cuda/bin/python}"
command -v docker >/dev/null
if command -v zstd >/dev/null 2>&1; then
    COMPRESS=(zstd -T0)
else
    # The copied evaluator environment already has zstandard; require no host install.
    COMPRESS=(docker run --rm -i --network none "$IMAGE" /submission/runtime/eval-env/bin/python -B -c
        'import sys,zstandard; zstandard.ZstdCompressor(level=3,threads=0).copy_stream(sys.stdin.buffer,sys.stdout.buffer)')
fi
[[ -d "$DATASET_ROOT" ]]
nice -n 19 "$PYTHON" -B scripts/container/stage.py --finalize
[[ -f "$BUILD/submission/submission.json" ]]
mkdir -p "$BUILD/docker-results"
printf '%s\n' 'Building pulls ubuntu:26.04 (unless BASE_IMAGE overridden), and the apt packages listed in scripts/container/Dockerfile. Python packages and models are copied, never installed.'
docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$IMAGE" -f scripts/container/Dockerfile "$BUILD" 2>&1 | tee "$BUILD/docker-build.log"
docker run --rm --network none "$IMAGE" /submission/run.sh verify-copies
if [[ "${ORGANISER_ACCEPTS_TWO_COMMITS:-0}" == 1 ]]; then
    # Use only after organisers explicitly accept two independent manifests/checkouts.
    # The aggregate manifest still fails their current single-commit implementation.
    for label in poses-combined-v5 meshes-d1-v5; do
        docker run --rm --network none -v "$DATASET_ROOT:/data:ro" -v "$BUILD/docker-results:/output" \
            -e "MANIFEST=/submission/$label.json" "$IMAGE" /submission/run.sh verify \
            --dataset-root /data --report "/output/$label-verifier.json" 2>&1 | tee "$BUILD/$label-verifier.log"
    done
else
    docker run --rm --network none -v "$DATASET_ROOT:/data:ro" -v "$BUILD/docker-results:/output" \
        "$IMAGE" /submission/run.sh verify --dataset-root /data --report /output/aggregate-verifier.json \
        2>&1 | tee "$BUILD/aggregate-verifier.log"
fi
docker run --rm --network none -v "$BUILD/docker-results:/output" "$IMAGE" \
    /submission/run.sh verify-mesh --manifest /submission/split.draft.json --output /output 2>&1 | tee "$BUILD/mesh-repack.log"
docker run --rm --gpus all --network none -v "$DATASET_ROOT:/data:ro" -v "$BUILD/docker-results:/output" \
    "$IMAGE" /submission/run.sh reproduce --label poses-combined-v5 --dataset-root /data --output /output \
    2>&1 | tee "$BUILD/reproduce.log"
docker run --rm --network none -v "$BUILD/docker-results:/output" \
    -e MANIFEST=/submission/poses-combined-v5.json "$IMAGE" /submission/run.sh verify \
    --reproduced poses-combined-v5=/output/poses-combined-v5/track3_combined_v5.parquet \
    --report /output/reproduced-verifier.json 2>&1 | tee "$BUILD/reproduced-verifier.log"
docker save "$IMAGE" | "${COMPRESS[@]}" > "$BUILD/video2data-submission.tar.zst"
sha256sum "$BUILD/video2data-submission.tar.zst" | tee "$BUILD/tarball.sha256"
docker image inspect "$IMAGE" --format '{{.Id}}' | tee "$BUILD/image-id.txt"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}' | tee "$BUILD/repo-digests.json"
cat <<EOF
Save and reproduction checks completed. Read docker-results and compare reproduced scores with the counted pose scores before submitting the form.
Upload by hand: $BUILD/video2data-submission.tar.zst
Paste into the form: public download link, checksum in tarball.sha256, and image ID in image-id.txt.
This locally built image may have no registry RepoDigest; image-id.txt is its content-addressed config ID, not a registry manifest digest. Ask organisers if their digest field specifically requires the latter.
No upload, push or form submission was performed.
EOF
"$PYTHON" -B - "$BUILD/submission/submission.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
urls = {m['code']['commit_url']} | {e['code_commit_url'] for e in m['submissions'] if 'code_commit_url' in e}
print('Counted code URL(s):\n' + '\n'.join(sorted(urls)))
PY
