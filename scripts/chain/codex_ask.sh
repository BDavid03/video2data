#!/usr/bin/env bash
# Ask Codex one question, or hand it one bounded job, the same way from every chain context.
#
#   bash scripts/chain/codex_ask.sh <label> <brief.md> [mode] [model] [effort]
#     label   short name, letters/digits/dashes (e.g. mesh-ideas, critic-03)
#     brief   a Markdown file holding the whole request (Codex starts with no context)
#     mode    read-only        ideas, reviews, critique (default)
#             workspace-write  a job on the paths the brief names; sandboxed, CPU only (the
#                              sandbox blocks the GPU on this WSL machine)
#             gpu              a job that needs the GPU; runs WITHOUT the sandbox
#                              (danger-full-access, approved by the owner 2026-11-02). Only one
#                              gpu job at a time: a second one exits 75 at once.
#             The calling context reviews `git diff` and commits; Codex never commits.
#     model   default gpt-6-astra for read-only, gpt-6.1-sol for the two job modes
#     effort  default high
#
#   CODEX_RESUME=<earlier answer.md or session id>  sends the brief as a follow-up in that
#     Codex session (it keeps everything it read and did), e.g. "fix these review defects".
#   CODEX_TIMEOUT_MIN  cap in minutes (default 60 read-only, 180 for job modes).
#
# Writes outputs/chain/codex/<UTC>-<label>.{brief.md,answer.md,log} and appends one line to
# outputs/chain/codex/calls.jsonl (with the session id). The last line printed is
# "CODEX <rc> <answer path>". Run it with run_in_background and end the turn; never poll it.
set -u
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 64
label=${1:?label}; brief=${2:?brief file}; mode=${3:-read-only}
case "$mode" in
  read-only)       model=${4:-gpt-6-astra};  sandbox=read-only;          minutes=60 ;;
  workspace-write) model=${4:-gpt-6.1-sol};  sandbox=workspace-write;    minutes=180 ;;
  gpu)             model=${4:-gpt-6.1-sol};  sandbox=danger-full-access; minutes=180 ;;
  *) echo "mode must be read-only, workspace-write or gpu" >&2; exit 64 ;;
esac
effort=${5:-high}
minutes=${CODEX_TIMEOUT_MIN:-$minutes}
[[ "$label" =~ ^[A-Za-z0-9-]+$ ]] || { echo "label: letters, digits and dashes only" >&2; exit 64; }
[ -f "$brief" ] || { echo "no brief file: $brief" >&2; exit 64; }
codex=$(command -v codex || ls -1 "$HOME"/.nvm/versions/node/*/bin/codex 2>/dev/null | tail -1)
[ -n "$codex" ] || { echo "codex CLI not found" >&2; exit 64; }

dir=outputs/chain/codex; mkdir -p "$dir"
if [ "$mode" = gpu ]; then
  exec 9>"$dir/gpu.lock"
  flock -n 9 || { echo "another Codex gpu job holds $dir/gpu.lock; wait for it" >&2; exit 75; }
fi

session=""
if [ -n "${CODEX_RESUME:-}" ]; then
  session=$CODEX_RESUME
  if [ -f "$session" ]; then
    session=$(grep -m1 -oP '^session id: \K\S+' "${session%.answer.md}.log" 2>/dev/null)
    [ -n "$session" ] || { echo "no session id in ${CODEX_RESUME%.answer.md}.log" >&2; exit 64; }
  fi
fi

stem="$dir/$(date -u +%Y%m%dT%H%M%SZ)-$label"
{
  if [ -n "$session" ]; then
    echo "--- FOLLOW-UP (same rules as before; mode now: $mode) ---"
  else
    cat <<'EOF'
You are collaborating with a Claude Code context on the video2data Track 3 challenge entry in
/home/ceolwolf/repos/video2data. You do most of the hands-on work in this chain: the Claude
context plans, briefs you, reviews your diff, re-checks your numbers and commits.
AGENTS.md, CLAUDE.md and docs/chain/PROGRAM.md hold the rules; read the parts the request below
points to. Rules that always bind you:
- Never rm -rf or shutil.rmtree anything except a directory you created yourself under .runtime/tmp/.
- Run no mutating git command (no add, commit, checkout, reset, stash or clean).
- Never stop or start a process you did not start; never kill anything.
- Never read or write /mnt/z. Never use holdout episodes 43/44 or episode 41.
- Write only inside /home/ceolwolf/repos/video2data: the paths the request names, plus outputs/,
  .runtime/tmp/ and .cache/. The dataset snapshot, upstream/ and models/ are read-only.
  List every file you changed.
- GPU work only when the request is in gpu mode, one operation at a time, through the slice
  handlers and the lease in track3/gpu.py (never touch a lock a live process owns). A run you
  expect to take longer than about two hours: start it detached
  (setsid nohup ... > outputs/<slice>/<name>.log 2>&1 &) and report its PID, log and how to tell
  it has finished, instead of waiting.
- Install nothing. Upload nothing. No test suites: verify with py_compile, a CPU import, an
  old-versus-new equivalence script, and real runs on real data.
How to build (docs/chain/PROGRAM.md section 9): code runs the loop, agents steer it. For
repeated trials (sweeps, tuning, ablations, seeds) build a resumable loop script once: it reads
an editable space/queue file, runs each trial through the slice handlers, skips finished trials,
appends a ledger, and ends with a compact ranked summary to read; do not run trials one by one
yourself. Within trials use existing code (NumPy/SciPy/OpenCV/torch, track3/, the kit); do not
over-engineer (smallest thing that works, no abstraction before its second use); keep library
code (pure functions, no I/O, paths, gates or run records) apart from the slice handler that
orchestrates them; keep each feature in its vertical slice track3/features/<slice>/; write clean,
presentable code and a README that says how to run it and what it found.
Be concrete: cite files with line numbers, give numbers, say what you checked and what you only
believe. Disagree when the evidence says so. End with: files changed, commands run, results
(numbers and artifact paths), and anything left undone.

--- REQUEST ---
EOF
  fi
  cat "$brief"
} > "$stem.brief.md"

common=(-m "$model" -c "model_reasoning_effort=\"$effort\"" -c "sandbox_mode=\"$sandbox\""
        --skip-git-repo-check -o "$stem.answer.md")
start=$(date -u +%FT%TZ)
if [ -n "$session" ]; then
  timeout "$((minutes * 60))" "$codex" exec resume "${common[@]}" "$session" - < "$stem.brief.md" > "$stem.log" 2>&1
else
  timeout "$((minutes * 60))" "$codex" exec "${common[@]}" -C "$PWD" --color never - < "$stem.brief.md" > "$stem.log" 2>&1
fi
rc=$?
sid=$(grep -m1 -oP '^session id: \K\S+' "$stem.log" 2>/dev/null)
python3 -c 'import json,sys; print(json.dumps(dict(zip(["start","label","mode","model","effort","rc","answer","session"], sys.argv[1:]))))' \
  "$start" "$label" "$mode" "$model" "$effort" "$rc" "$stem.answer.md" "${sid:-$session}" >> "$dir/calls.jsonl"
[ -s "$stem.answer.md" ] || echo "no answer written; see $stem.log (usage limit, auth or timeout)"
echo "CODEX $rc $stem.answer.md"
exit "$rc"
