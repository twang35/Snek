#!/bin/bash
# Waits for a policy's first checkpoint, then execs into watch.py.
#
#   cd snek2
#   ./watch_when_ready.sh <policy_name>
#
# watch.py exits immediately (code 1) if savedPolicies/<policy_name>/ has no checkpoint yet,
# which is the normal state for a freshly-launched arm: training skips writing checkpoints
# until the score clears SNEK_MIN_CHECKPOINT_SCORE (default 40). Launching watch.py itself
# right after `snek2.py <policy>` therefore usually just exits — this polls until the first
# checkpoint lands and then hands off to watch.py, which does its own re-checking after that
# for every later checkpoint. Safe to launch immediately alongside the trainer; it costs
# nothing while it waits.
set -u
POLICY_NAME="${1:?usage: watch_when_ready.sh <policy_name>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$HERE/savedPolicies/$POLICY_NAME"
POLL_SECONDS="${WATCH_WHEN_READY_POLL:-30}"

cd "$HERE" || exit 1
while true; do
  if ls "$CKPT_DIR"/ckpt-*.index >/dev/null 2>&1; then
    echo "checkpoint found for $POLICY_NAME, attaching watch.py"
    exec env PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u watch.py "$POLICY_NAME"
  fi
  sleep "$POLL_SECONDS"
done
