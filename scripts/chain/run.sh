#!/usr/bin/env bash
# Start the chain driver in a tmux session named "chain" (attach with: tmux attach -t chain).
# Its output is also appended to data/outputs/chain/driver.log, and the window stays open after the
# driver exits so the reason can be read. Extra arguments go to session_chain.py.
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
mkdir -p data/outputs/chain
if [ "$1" != "--here" ]; then
  if tmux has-session -t chain 2>/dev/null; then
    echo "a tmux session 'chain' already exists: tmux attach -t chain"; exit 1
  fi
  exec tmux new -s chain "bash scripts/chain/run.sh --here${*:+ $(printf '%q ' "$@")}"
fi
shift
echo "[$(date '+%F %T')] run.sh: python3 scripts/chain/session_chain.py $*" >> data/outputs/chain/driver.log
python3 scripts/chain/session_chain.py "$@" 2>&1 | tee -a data/outputs/chain/driver.log
rc=${PIPESTATUS[0]}
echo "driver exited rc=$rc (full output: data/outputs/chain/driver.log). Press Enter to close this window."
read -r
