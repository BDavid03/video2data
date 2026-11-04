# Dedicated policy environment

Status: implemented, never run

These scripts are written preparation, not evidence of an installation. Do not
run them under the current no-execution instruction. The existing Ubuntu 26.04
distribution is never selected, installed into, made default or unregistered.

When separately authorized by the user:

1. Supply an Ubuntu 24.04 WSL rootfs tarball inside this project. Review
   `configs/runtime/policy-env.json`, authorize creation there, then run
   `create-policy-distro.ps1 -RootfsTarball <path> -Confirm`. Both the flag and
   switch are required. The script refuses an existing distribution or target
   directory and imports only `video2data-policy` as WSL 2. Its VHDX lives in
   `.runtime/wsl/video2data-policy`; it does not start or provision it.
2. Separately authorize installation in the config. In that distribution only,
   as root, run `bash /mnt/z/hugging-face/video2data/scripts/wsl/install-policy-env.sh`.
   It refuses other distributions, other Ubuntu versions and existing installs.
   A failed installation stays in place with a failed record for manual review.
3. Dot-source `scripts/use-policy-env.ps1` and explicitly invoke
   `Enter-Video2DataPolicy -Environment policy` or `-Environment retarget`.
   Alternatively source `scripts/wsl/use-policy-env.sh policy` or `retarget`
   inside the dedicated distribution. Activation itself runs no Python.

The installer checks out the full pinned commit with LF bytes and fetches robot
Git LFS assets inside `/opt/video2data` on the project-owned WSL disk. It never
installs from or modifies the Windows upstream reference. FlashCHORD uses Python
3.11 and `uv sync --locked --extra train` against its own lock. The kit is copied
unchanged after checking every manifest entry. The installation record includes
the source commit and evaluator digest using sorted module names, a NUL after
each name, and raw module bytes, exactly as upstream `harness.py` specifies.

Retargeting has a separate Python 3.11 environment on the same disk: its Dockerfile
requires NumPy below 2, incompatible with FlashCHORD's NumPy 2.4.6. The requirement
file records the Dockerfile's retargeting native closure and supporting imports.
Isaac Lab, whole-body SOMA and tests are not installed. Torch and the other
unversioned requirements inherited from the Docker base have no source-confirmed
standalone version; the installer records the resolved retargeting package list.
No dependency resolution has been attempted. Both environments use project-local
cache/temp paths. The WSL disk contains the Linux source and environments; output
artifacts remain under `/mnt/z/hugging-face/video2data`.

Creating the distribution, installing dependencies and executing any slice are
separate authorizations. All supplied flags are false. No episode is run by
these scripts.
