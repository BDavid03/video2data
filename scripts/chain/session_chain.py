"""Chain fresh Claude Code background sessions through hand-off files, with no copy-paste.

Each context reads docs/chain/PROGRAM.md and the newest hand-off in docs/chain/handoffs/, does one or two
measured improvements, and ends by writing the next hand-off. That file's last section is
"## Prompt for the next context (paste as the first message)". This driver does the paste. It starts the next
context as a fresh `claude --bg` session with that prompt, word for word, waits for it to write its own hand-off
and go quiet, and repeats. Every context still starts clean from a hand-off written by the one before, so the
quality of the chain is unchanged; only the manual step is gone. On every launch the driver also appends a short
note to the system prompt: the real clock, the hours left to the leaderboard freeze, and a pointer to PROGRAM.md.
That way the standing rules come back every time, whatever the hand-off remembered to copy.

Usage (in WSL, from the checkout):
  python3 scripts/chain/session_chain.py --dry-run     # print the next launch and exit
  python3 scripts/chain/session_chain.py               # run the chain in this terminal
  nohup python3 scripts/chain/session_chain.py > outputs/chain/driver.log 2>&1 &   # or in tmux
  python3 scripts/chain/session_chain.py --max-contexts 6       # every context runs on Opus unless --model says

The chain stops, with the reason on the last line, when:
  COMPLETE  the newest hand-off has a `CHAIN: STOP <reason>` line (the context says the work is finished, or that
            the next context could do nothing useful without the owner)
  LIMIT     a context is blocked by spent usage (credits or a rate limit); the job detail is printed
  BUDGET    --max-contexts contexts were launched, or the jobs this run launched reached --token-budget
  STALLED   a context sat idle with nothing in flight and no new hand-off for --stall-minutes, a hand-off has no
            prompt section, or two contexts in a row ended inside --min-minutes
  STOPPED   the file outputs/chain/STOP exists
The file outputs/chain/PAUSE holds the next launch until it is deleted; use it to read or edit a hand-off first.

A blocked context that is not a usage limit (for example a permission prompt) is reported and waited on: answer it
with `claude attach <id>` and the chain carries on.

Standard library only. It writes only outputs/chain/; it never stops a session or a process.
Exit codes: 0 COMPLETE, 2 BUDGET, 3 LIMIT, 4 STALLED, 5 STOPPED.
"""
import argparse, datetime, glob, json, os, re, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADING = re.compile(r"^## Prompt for the .*\(paste as the first message\)[ \t]*\r?$", re.M)
STOP_LINE = re.compile(r"^\s*CHAIN:\s*STOP\b[ \t]*(.*)$", re.M)
LIMIT_WORDS = re.compile(r"usage|credit|rate.?limit|hit your limit", re.I)
EXIT = {"COMPLETE": 0, "BUDGET": 2, "LIMIT": 3, "STALLED": 4, "STOPPED": 5}

CHAIN_NOTE = """\
CHAIN CONTEXT. The chain driver (scripts/chain/session_chain.py) started this session and will start the next one:
nobody is pasting prompts. Clock at launch: {now_utc} UTC ({now_local} local). Leaderboard freeze: {deadline_utc} UTC,
{hours_left} hours from launch. Context number {number} of this run.
1. Before anything else read docs/chain/PROGRAM.md. It binds every context, together with AGENTS.md and CLAUDE.md.
   Where your prompt and PROGRAM.md disagree, PROGRAM.md wins; record the disagreement in your hand-off.
2. Hand off exactly as PROGRAM.md says: a new file docs/chain/handoffs/HANDOFF-<UTC yyyymmddTHHMMZ>-<NN>.md whose
   last section is "## Prompt for the next context (paste as the first message)" with the complete next prompt.
   Write that file last, in one write, once this context is finished with: its appearance starts the next context.
3. If the work is complete, or the next context could do nothing useful without the owner, put a line
   `CHAIN: STOP <one-line reason>` in that hand-off. The chain then ends and the owner reads it.
4. Nobody is watching: never wait for an answer. Owner-only items go in docs/chain/NEEDS-YOU.md; carry on with
   other work.
5. Never copy this note into the hand-off prompt; the driver adds it to every context."""


def now():
    return time.time()


def stamp():
    return datetime.datetime.now().strftime("%H:%M")


def find_claude():
    found = shutil.which("claude")
    if found:
        return found
    nvm = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/*/bin/claude")))
    return nvm[-1] if nvm else "claude"


class Chain:
    def __init__(self, a):
        self.a = a
        self.jobs_dir = os.path.expanduser("~/.claude/jobs")
        self.handoff_dir = os.path.join(ROOT, a.handoffs)
        self.log_dir = os.path.join(ROOT, a.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "chain.jsonl")
        pre, _, post = a.name_format.partition("{n}")
        self.name_re = re.compile(r"^\W*" + re.escape(pre) + r"(\d+)" + re.escape(post) + r"$")
        self.deadline = datetime.datetime.fromisoformat(a.deadline.replace("Z", "+00:00"))
        self.launched = []  # job ids this run started

    # ---- output -------------------------------------------------------------------------------------------
    def say(self, text, **event):
        print(f"[{stamp()}] {text}", flush=True)
        if event:
            event["at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def finish(self, reason, text):
        self.say(f"{reason}: {text}", event="end", reason=reason, text=text, launched=self.launched)
        print(f"chain {reason}: {text}", flush=True)
        sys.exit(EXIT[reason])

    # ---- hand-offs ----------------------------------------------------------------------------------------
    def newest_handoff(self):
        """(path, text, age_seconds) of the newest hand-off. Names sort by their UTC stamp, so the name decides."""
        files = sorted(glob.glob(os.path.join(self.handoff_dir, "HANDOFF-*.md")))
        if not files:
            raise RuntimeError(f"no HANDOFF-*.md in {self.handoff_dir}")
        path = files[-1]
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        return path, text, now() - os.path.getmtime(path)

    @staticmethod
    def prompt_of(text):
        m = HEADING.search(text)
        if not m:
            return None
        rest = text[m.end():]
        nxt = re.search(r"^## ", rest, re.M)
        return (rest[:nxt.start()] if nxt else rest).strip()

    # ---- jobs ---------------------------------------------------------------------------------------------
    def jobs(self):
        found = []
        for f in glob.glob(os.path.join(self.jobs_dir, "*", "state.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            d["_id"] = os.path.basename(os.path.dirname(f))
            found.append(d)
        return found

    def job(self, job_id):
        try:
            with open(os.path.join(self.jobs_dir, job_id, "state.json"), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return None
        d["_id"] = job_id
        return d

    def chain_jobs(self):
        """Jobs named by the chain's name format, as (number, job), oldest first."""
        out = []
        for d in self.jobs():
            m = self.name_re.match(d.get("name") or "")
            if m:
                out.append((int(m.group(1)), d))
        return sorted(out, key=lambda x: (x[0], x[1].get("createdAt", "")))

    def consumer(self, prompt):
        """The newest job whose first message was this prompt (the job list keeps it verbatim as `intent`).
        A job a usage limit blocked before it did any work never started, so its prompt is launched again."""
        hits = [d for d in self.jobs() if (d.get("intent") or "").strip() == prompt and not self.never_started(d)]
        return max(hits, key=lambda d: d.get("createdAt", "")) if hits else None

    def never_started(self, d):
        return (d.get("state") == "blocked" and self.tokens(d) == 0
                and bool(LIMIT_WORDS.search(d.get("detail") or "")))

    @staticmethod
    def busy(d):
        """Mid-turn: still writing or running a tool."""
        return d.get("state") == "working" and d.get("tempo") != "idle"

    @staticmethod
    def tasks(d):
        return int((d.get("inFlight") or {}).get("tasks") or 0)

    @staticmethod
    def tokens(d):
        try:
            return int(d.get("tokens") or 0)
        except ValueError:
            return 0

    # ---- launch -------------------------------------------------------------------------------------------
    def previous_session(self):
        """The last context of this chain, whose flags carry forward (the model is pinned by --model)."""
        chain = self.chain_jobs()
        return (chain[-1][0], chain[-1][1]) if chain else (0, None)

    def launch_plan(self, prompt):
        last_number, prev = self.previous_session()
        number = last_number + 1
        flags = self.a.flags.split() if self.a.flags else self.copy_flags(prev)
        flags = self.override(flags, "--model", self.a.model)
        flags = self.override(flags, "--effort", self.a.effort)
        name = self.a.name_format.replace("{n}", f"{number:02d}")
        utc = datetime.datetime.now(datetime.timezone.utc)
        note = CHAIN_NOTE.format(
            now_utc=utc.strftime("%Y-%m-%d %H:%M"), now_local=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            deadline_utc=self.deadline.strftime("%Y-%m-%d %H:%M"),
            hours_left=f"{(self.deadline - utc).total_seconds() / 3600:.1f}", number=len(self.launched) + 1)
        argv = [self.a.claude, "--bg", *flags, "--name", name, "--append-system-prompt", note, prompt]
        return name, flags, argv, prev

    @staticmethod
    def copy_flags(prev):
        if not prev or not prev.get("respawnFlags"):
            return ["--agent", "claude", "--permission-mode", "auto", "--model", "opus"]
        out, it = [], iter(prev["respawnFlags"])
        for flag in it:
            if flag in ("--name", "-n", "--append-system-prompt"):
                next(it, None)
            elif flag not in ("--bg", "--background"):
                out.append(flag)
        return out

    @staticmethod
    def override(flags, flag, value):
        if not value:
            return flags
        out, it = [], iter(flags)
        for f in it:
            if f == flag:
                next(it, None)
            else:
                out.append(f)
        return out + [flag, value]

    def launch(self, path, prompt):
        name, flags, argv, _ = self.launch_plan(prompt)
        r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
        m = re.search(r"backgrounded\s*\W\s*([0-9a-f]{8})", r.stdout)
        if r.returncode != 0 or not m:
            raise RuntimeError(f"claude --bg failed rc={r.returncode}: {(r.stdout + r.stderr).strip()[:600]}")
        job_id = m.group(1)
        self.launched.append(job_id)
        self.say(f"LAUNCHED {name} ({job_id}) from {os.path.basename(path)} [{' '.join(flags)}]; "
                 f"watch: claude attach {job_id}",
                 event="launch", job=job_id, name=name, handoff=os.path.basename(path), flags=flags)
        deadline = now() + 120
        while self.job(job_id) is None and now() < deadline:
            time.sleep(2)
        return job_id

    # ---- watch --------------------------------------------------------------------------------------------
    def check_stop(self):
        if os.path.exists(os.path.join(self.log_dir, "STOP")):
            self.finish("STOPPED", f"{self.log_dir}/STOP exists")

    def wait_pause(self):
        paused = False
        while os.path.exists(os.path.join(self.log_dir, "PAUSE")):
            if not paused:
                self.say(f"PAUSED: delete {self.log_dir}/PAUSE to continue", event="pause")
                paused = True
            time.sleep(15)
            self.check_stop()
        return paused

    def watch(self, job_id, handoff_path):
        """Wait until the context writes a newer hand-off and goes quiet. Returns the job's final state."""
        a = self.a
        last_line, seen_new_at, quiet_since, blocked_said = None, None, None, False
        while True:
            self.check_stop()
            d = self.job(job_id)
            if d is None:
                self.finish("STALLED", f"job {job_id} vanished from {self.jobs_dir}")
            name, detail = d.get("name"), (d.get("detail") or "").replace("\n", " ")
            line = (f"{name} {d.get('state')}/{d.get('tempo')} tasks={self.tasks(d)} "
                    f"tokens={self.tokens(d):,} :: {detail[:160]}")
            if line != last_line:
                self.say(line)
                last_line = line

            if d.get("state") == "blocked":
                if LIMIT_WORDS.search(detail):
                    self.finish("LIMIT", f"{name} ({job_id}) blocked: {detail[:300]} -- resume it with "
                                         f"`claude attach {job_id}` once usage is back, then rerun this driver")
                if not blocked_said:
                    self.say(f"BLOCKED {name}: {detail[:300]} -- answer it with `claude attach {job_id}`",
                             event="blocked", job=job_id, detail=detail)
                    blocked_said = True
                time.sleep(a.poll)
                continue
            blocked_said = False

            newest, _, age = self.newest_handoff()
            quiet = not self.busy(d)
            if newest != handoff_path:
                seen_new_at = seen_new_at or now()
                settled = age >= a.settle and now() - seen_new_at >= a.settle
                tasks_ok = self.tasks(d) == 0 or now() - seen_new_at >= a.grace_minutes * 60
                if quiet and settled and tasks_ok:
                    self.say(f"HANDED OFF {name} -> {os.path.basename(newest)}", event="handoff", job=job_id,
                             handoff=os.path.basename(newest), tokens=self.tokens(d))
                    return d
            elif quiet and self.tasks(d) == 0:
                quiet_since = quiet_since or now()
                if now() - quiet_since >= a.stall_minutes * 60:
                    self.finish("STALLED", f"{name} ({job_id}) has been idle {a.stall_minutes} min with nothing in "
                                           f"flight and no new hand-off; read it with `claude attach {job_id}`")
            else:
                quiet_since = None
            time.sleep(a.poll)

    # ---- main loop ----------------------------------------------------------------------------------------
    def run(self):
        a, short_runs = self.a, 0
        self.say(f"chain {a.name_format} over {self.handoff_dir}; log {self.log_path}", event="start", args=vars(a))
        while True:
            self.check_stop()
            path, text, age = self.newest_handoff()
            base = os.path.basename(path)
            stop = STOP_LINE.search(text)
            if stop:
                self.finish("COMPLETE", f"{base}: {stop.group(1).strip() or 'CHAIN: STOP'}")
            prompt = self.prompt_of(text)
            if prompt is None:
                if age < a.stall_minutes * 60:  # still being written, or written without the section: give it time
                    time.sleep(a.poll)
                    continue
                self.finish("STALLED", f"{base} has no '## Prompt for the ... (paste as the first message)' section")
            if not prompt:
                self.finish("STALLED", f"{base} has an empty prompt section")

            job = self.consumer(prompt)
            if job is None:
                if a.dry_run:
                    name, flags, _, prev = self.launch_plan(prompt)
                    prev = prev or {}
                    print(f"would launch {name} in {ROOT} with {' '.join(flags)}")
                    print(f"  after: {prev.get('name')} ({prev.get('_id')}) state={prev.get('state')}/"
                          f"{prev.get('tempo')} tasks={self.tasks(prev)}")
                    print(f"  hand-off: {base}; prompt {len(prompt)} chars:")
                    print("  | " + "\n  | ".join(prompt.splitlines()[:4]) + "\n  | ...")
                    return 0
                if len(self.launched) >= a.max_contexts:
                    self.finish("BUDGET", f"{a.max_contexts} contexts launched this run; next is {base}")
                spent = sum(self.tokens(self.job(j) or {}) for j in self.launched)
                if a.token_budget and spent >= a.token_budget:
                    self.finish("BUDGET", f"jobs launched this run report {spent:,} tokens >= {a.token_budget:,}")
                if self.wait_pause():
                    continue  # re-read the hand-off: it may have been edited during the pause
                if age < a.settle:
                    time.sleep(a.settle - age)
                    continue  # re-read once the file has stopped changing
                job_id = self.launch(path, prompt)
            else:
                job_id = job["_id"]
                if a.dry_run:
                    print(f"{base} is already running as {job.get('name')} ({job_id}) state={job.get('state')}/"
                          f"{job.get('tempo')} tasks={self.tasks(job)}: the driver would watch it")
                    return 0
                self.say(f"resuming watch of {job.get('name')} ({job_id}), started from {base}")

            final = self.watch(job_id, path)
            try:
                started = datetime.datetime.fromisoformat(final["createdAt"].replace("Z", "+00:00")).timestamp()
                ran = (now() - started) / 60
            except (KeyError, AttributeError, ValueError):
                ran = a.min_minutes
            short_runs = short_runs + 1 if ran < a.min_minutes else 0
            if short_runs >= 2:
                self.finish("STALLED", f"two contexts in a row handed off inside {a.min_minutes} min; read the "
                                       f"newest hand-off before restarting")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--handoffs", default="docs/chain/handoffs", help="folder of HANDOFF-*.md, relative to the checkout")
    p.add_argument("--name-format", default="{n}-iter-video2data", help="session name; {n} is the number")
    p.add_argument("--deadline", default="2026-11-04T22:00:00Z",
                   help="leaderboard freeze in UTC (November 4, 2026 at 5:00 p.m. EST)")
    p.add_argument("--model", default="opus", help="model for every context (default opus, the owner's choice)")
    p.add_argument("--effort", help="override the effort copied from the previous context")
    p.add_argument("--flags", help="replace the copied flags entirely (one string, e.g. '--agent claude --model opus')")
    p.add_argument("--max-contexts", type=int, default=30, help="contexts to launch this run (default 30)")
    p.add_argument("--token-budget", type=int, default=0,
                   help="stop launching once this run's jobs report this many tokens")
    p.add_argument("--min-minutes", type=float, default=15, help="a context ending sooner counts as a short run")
    p.add_argument("--stall-minutes", type=float, default=45, help="idle with nothing in flight and no hand-off")
    p.add_argument("--grace-minutes", type=float, default=20,
                   help="after a hand-off, launch even if the old context still has background tasks after this long")
    p.add_argument("--settle", type=float, default=90, help="seconds a new hand-off must stay unchanged")
    p.add_argument("--poll", type=float, default=60, help="seconds between checks")
    p.add_argument("--log-dir", default="outputs/chain", help="chain.jsonl, PAUSE and STOP live here")
    p.add_argument("--claude", default=find_claude(), help="claude executable")
    p.add_argument("--dry-run", action="store_true", help="print the next launch and exit")
    a = p.parse_args()
    try:
        sys.exit(Chain(a).run() or 0)
    except KeyboardInterrupt:
        print("interrupted; running sessions are untouched, rerun to resume the watch", flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
