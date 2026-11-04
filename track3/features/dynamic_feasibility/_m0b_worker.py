"""Owned upstream runner with timestamped progress and cooperative M0b stops."""

import json
import os
from pathlib import Path
import runpy
import sys
import time


class BudgetStop(Exception):
    """End an owned training child at a completed collection boundary."""


def main():
    script, *arguments = sys.argv[1:]
    origin = time.monotonic()
    deadline = float(os.environ['DYNAMIC_FEASIBILITY_DEADLINE'])
    operation = os.environ['M0B_OPERATION']
    output = Path(os.environ['M0B_WORKER_RECORD'])
    last, full_at = None, None
    stop_reason = 'completed'
    from flash_chord.training import progress
    original = progress.ConsoleTrainingProgress

    class Clock(original):
        def start(self, completed_environment_steps):
            nonlocal last
            last = {'steps': completed_environment_steps, 'updates': 2632, 'replay_size': 0,
                    'credit': .75, 'seconds': time.monotonic() - origin}
            print(f"M0B t={last['seconds']:.9f} updates=2632 replay=0 credit=0.75 ", end='')
            super().start(completed_environment_steps)
            self.last_report = time.monotonic()
            if time.time() >= deadline:
                raise BudgetStop('deadline')

        def update(self, completed_environment_steps):
            nonlocal last, full_at
            now = time.monotonic()
            learner = sys._getframe(1).f_locals['learner']
            expected_full_steps = 21575680 + int(os.environ['M0B_CAPACITY']) + 2048
            first_full = full_at is None and completed_environment_steps >= expected_full_steps
            stopping = time.time() >= deadline or (operation == 'probe' and full_at is not None
                                                    and now - origin >= full_at + 60)
            if now - self.last_report >= 1 or first_full or stopping or completed_environment_steps == self.total_environment_steps:
                import jax
                updates, size, credit = jax.device_get((learner.global_update_step, learner.replay.size, learner.update_credit))
                last = {'steps': completed_environment_steps, 'updates': int(updates), 'replay_size': int(size),
                        'credit': float(credit), 'seconds': time.monotonic() - origin}
                if int(size) == int(os.environ['M0B_CAPACITY']) and full_at is None:
                    full_at = last['seconds']
                print(f"M0B t={last['seconds']:.9f} updates={int(updates)} replay={int(size)} credit={float(credit):.9f} ", end='')
                self._next_percent = 0
                super().update(completed_environment_steps)
                self.last_report = time.monotonic()
            if stopping:
                raise BudgetStop('deadline' if time.time() >= deadline else 'ring_full_60_seconds')

    progress.ConsoleTrainingProgress = Clock
    calls = 0

    def setup_deadline(frame, event, arg):
        nonlocal calls
        calls += 1
        if calls % 4096 == 0 and last is None and time.time() >= deadline:
            raise BudgetStop('deadline_during_setup')

    sys.argv = [script, *arguments]
    sys.setprofile(setup_deadline)
    try:
        runpy.run_path(script, run_name='__main__')
    except BudgetStop as error:
        stop_reason = str(error)
    except BaseException as error:
        stop_reason = f'error:{type(error).__name__}'
        raise
    finally:
        sys.setprofile(None)
        output.write_text(json.dumps({'reason': stop_reason, 'last_progress': last,
                                      'ring_full_seconds': full_at, 'wall_seconds': time.monotonic() - origin}, indent=2) + '\n')
        print(f'M0B_STOP {stop_reason}', flush=True)


if __name__ == '__main__':
    main()
