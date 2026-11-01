# Automated context chain

A chain of fresh Claude Code sessions works on the Track 3 submission and keeps improving it.
Each session (a "context") reads the standing brief, does one or two measured improvements and
writes a hand-off. The driver then starts the next context from that hand-off, so nobody copies
prompts by hand.

| File | What it is |
| --- | --- |
| `PROGRAM.md` | The standing brief every context reads first: finish line, lenses, integrity and safety rules, the hand-off format |
| `SUBMISSION.md` | The deliverable checklist (D1-D6) and the upload log |
| `SCOREBOARD.md` | One score per lens (models, metrics, visualisations, presentation, insights, engineering, readiness), set by a fresh-context critic |
| `IDEAS.md` | The road-less-travelled ledger: every unconventional idea, its cheapest test and its result |
| `NEEDS-YOU.md` | What only the owner can do; grant an item with a dated line in `AGENTS.md` |
| `handoffs/HANDOFF-<UTC>-<NN>.md` | One per context; the last section is the next context's prompt |
| `../../scripts/chain/session_chain.py` | The driver |
| `../../scripts/chain/codex_ask.sh` | The one way a context calls Codex, which does most of the hands-on work: modes `read-only`, `workspace-write` (no GPU) and `gpu` (unsandboxed, one at a time), follow-ups via `CODEX_RESUME`; answers in `outputs/chain/codex/` |
| `../../outputs/chain/` | The driver's log (`chain.jsonl`) and its `PAUSE` / `STOP` control files |

## Running it

From a WSL shell in the checkout:

```bash
python3 scripts/chain/session_chain.py --dry-run     # shows the next launch, changes nothing
bash scripts/chain/run.sh                            # run it in tmux session "chain"; detach with Ctrl-b d
tmux attach -t chain                                 # look at it again later
```

`run.sh` keeps the window open after the driver exits and appends everything it prints to
`outputs/chain/driver.log`. Only one driver can run at a time (`outputs/chain/driver.lock`).

Every context runs on Opus (`--model` changes it). Other useful flags: `--max-contexts 6`,
`--token-budget 3000000`. Watch a context with `claude attach <id>` (the driver prints the id).

## Steering it

- `touch outputs/chain/PAUSE` holds the next launch; read or edit the newest hand-off, then delete
  the file.
- `touch outputs/chain/STOP` ends the chain at the next check. Running sessions are never
  stopped.
- Edit `PROGRAM.md` to change the brief; the next context reads it.
- Answer `NEEDS-YOU.md` with a dated line in `AGENTS.md`.

The chain ends on its own when a hand-off says `CHAIN: STOP`, when usage runs out, at the
budget, or when a context stalls. The reason is the last line the driver prints.
