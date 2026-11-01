#!/usr/bin/env bash
# Ask Codex one question, or hand it one bounded job, the same way from every chain context.
#
#   bash scripts/chain/codex_ask.sh <label> <brief.md> [mode] [model] [effort]
#     label   short name, letters/digits/dashes (e.g. mesh-ideas, critic-03)
#     brief   a Markdown file holding the whole request (Codex starts with no context)
#     mode    read-only (default: ideas, reviews, critique) or workspace-write (a bounded job
#             on the paths the brief names; the calling context reviews `git diff` and commits)
#     model   default gpt-6-astra for read-only, gpt-6.1-sol for workspace-write
#     effort  default high
#
# Writes outputs/chain/codex/<UTC>-<label>.{brief.md,answer.md,log} and appends one line to
# outputs/chain/codex/calls.jsonl. The last line printed is "CODEX <rc> <answer path>".
# Run it with run_in_background and end the turn; never poll it. A call is capped at 60 minutes.
set -u
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 64
label=${1:?label}; brief=${2:?brief file}; mode=${3:-read-only}
case "$mode" in
  read-only) model=${4:-gpt-6-astra} ;;
  workspace-write) model=${4:-gpt-6.1-sol} ;;
  *) echo "mode must be read-only or workspace-write" >&2; exit 64 ;;
esac
effort=${5:-high}
[[ "$label" =~ ^[A-Za-z0-9-]+$ ]] || { echo "label: letters, digits and dashes only" >&2; exit 64; }
[ -f "$brief" ] || { echo "no brief file: $brief" >&2; exit 64; }
codex=$(command -v codex || ls -1 "$HOME"/.nvm/versions/node/*/bin/codex 2>/dev/null | tail -1)
[ -n "$codex" ] || { echo "codex CLI not found" >&2; exit 64; }

dir=outputs/chain/codex; mkdir -p "$dir"
stem="$dir/$(date -u +%Y%m%dT%H%M%SZ)-$label"
{
  cat <<'EOF'
You are collaborating with a Claude Code context on the video2data Track 3 challenge entry in
/home/ceolwolf/repos/video2data. AGENTS.md, CLAUDE.md and docs/chain/PROGRAM.md hold the rules;
read the parts the request below points to. Rules that always bind you:
- Never rm -rf or shutil.rmtree anything except a directory you created yourself under .runtime/tmp/.
- Run no mutating git command (no add, commit, checkout, reset, stash or clean).
- Never stop or start a process you did not start; never kill anything.
- Never read or write /mnt/z. Never use holdout episodes 43/44 or episode 41.
- One GPU operation at a time: do not start GPU work unless the request says to, and then only through
  track3/gpu.py.
- Install nothing. Upload nothing.
- In workspace-write mode, edit only the paths the request names, and list every file you changed.
Be concrete: cite files with line numbers, give numbers, say what you checked and what you only
believe. Disagree when the evidence says so.

--- REQUEST ---
EOF
  cat "$brief"
} > "$stem.brief.md"

start=$(date -u +%FT%TZ)
timeout 3600 "$codex" exec -m "$model" -c "model_reasoning_effort=$effort" -s "$mode" -C "$PWD" \
  --skip-git-repo-check --color never -o "$stem.answer.md" - < "$stem.brief.md" > "$stem.log" 2>&1
rc=$?
python3 -c 'import json,sys; print(json.dumps(dict(zip(["start","label","mode","model","effort","rc","answer"], sys.argv[1:]))))' \
  "$start" "$label" "$mode" "$model" "$effort" "$rc" "$stem.answer.md" >> "$dir/calls.jsonl"
[ -s "$stem.answer.md" ] || echo "no answer written; see $stem.log (usage limit, auth or timeout)"
echo "CODEX $rc $stem.answer.md"
exit "$rc"
