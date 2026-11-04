#!/usr/bin/env python3
"""Run the closed research loop. Minimal CLI over track3.features.auto_research.

  python scripts/research_loop.py                 # loop with the feature's loop.json
  python scripts/research_loop.py -n 12 -H 3      # 12 iterations or 3 hours
  python scripts/research_loop.py --no-consult    # heuristic proposals only
  python scripts/research_loop.py --finalize      # freeze on tuning, assess holdout once
  python scripts/research_loop.py --report [--finalize-result outputs/.../result.json]

Re-executes itself with the project's Linux CUDA interpreter and the pinned
MoGe source trees on PYTHONPATH, so it can be started from any shell.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def reexec():
    config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8-sig"))
    python = ROOT / config["linux_runtime"]["python"]  # keep the venv symlink: resolving it would bypass site-packages
    if Path(sys.executable).resolve() == python.resolve():
        return
    if not python.exists():
        sys.exit(f"Linux CUDA interpreter missing: {python}\nRun scripts/linux/install-linux-cuda.sh first")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(ROOT / p) for p in ("", "upstream/moge", "upstream/moge-utils3d", "upstream/moge-pipeline"))
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    os.execve(str(python), [str(python), *sys.argv], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--iterations", type=int, help="max iterations this session")
    parser.add_argument("-H", "--hours", type=float, help="max wall hours this session")
    parser.add_argument("--patience", type=int, help="stop after this many iterations without improvement")
    parser.add_argument("--no-consult", action="store_true", help="skip Codex consultation; heuristic only")
    parser.add_argument("--no-baselines", action="store_true", help="skip the opening baseline queue")
    parser.add_argument("--finalize", action="store_true", help="freeze tuning selection and assess holdout once")
    parser.add_argument("--report", action="store_true", help="render the extended-abstract LaTeX report")
    parser.add_argument("--finalize-result", help="finalize result.json to include in the report")
    args = parser.parse_args()
    reexec()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from track3.features.auto_research.loop import LoopRequest, run_loop
    from track3.features.auto_research.finalize import FinalizeRequest, finalize
    from track3.features.auto_research.report import ReportRequest, write_report
    from track3.features.auto_research._common import FEATURE, read_json

    if not (args.finalize or args.report):
        values = read_json(FEATURE / "loop.json")
        if args.iterations: values["max_iterations"] = args.iterations
        if args.hours: values["max_hours"] = args.hours
        if args.patience: values["patience"] = args.patience
        if args.no_consult: values["consult"] = False
        if args.no_baselines: values["baselines"] = False
        print(run_loop(LoopRequest(**values)))
    if args.finalize:
        result = finalize(FinalizeRequest.from_file(FEATURE / "finalize.json"))
        print(result)
        args.finalize_result = str(result.relative_to(ROOT))
    if args.report:
        print(write_report(ReportRequest(finalize_result=args.finalize_result)))


if __name__ == "__main__":
    main()
