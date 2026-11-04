#!/usr/bin/env bash
# Exit at a given UTC time so a chain context can resume time-gated work when the harness wakes it.
#   bash scripts/chain/wait_until_utc.sh "2026-11-04 12:00"     (run in the background, then end the turn)
set -eu
target=$(date -u -d "${1:?UTC time}" +%s)
while [ "$(date -u +%s)" -lt "$target" ]; do sleep 30; done
date -u
