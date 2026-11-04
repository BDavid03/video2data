"""Dispatch the isolated, fixed FoundationPose pilot without importing GPU code here."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "data/.runtime/foundationpose/pilot_predict.py"), run_name="__main__")
