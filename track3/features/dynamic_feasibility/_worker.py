"""Run an unchanged upstream entry point with a cooperative wall-clock deadline."""

import os
import runpy
import sys
import time


def main():
    script, *arguments = sys.argv[1:]
    deadline = float(os.environ["DYNAMIC_FEASIBILITY_DEADLINE"])
    calls = 0

    def deadline_guard(frame, event, arg):
        nonlocal calls
        calls += 1
        if calls % 4096 == 0 and time.time() >= deadline:
            sys.setprofile(None)
            raise TimeoutError("Dynamic feasibility wall-clock budget exhausted")

    if time.time() >= deadline:
        raise TimeoutError("Dynamic feasibility deadline already passed")
    sys.argv = [script, *arguments]
    sys.setprofile(deadline_guard)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.setprofile(None)


if __name__ == "__main__":
    main()
