#!/bin/bash
# Wait for every running `eval_checkpoints.py` to finish, then run a launcher.
#
#   cd snek2
#   nohup bash hyperparamTuning/chain_after_evals.sh hyperparamTuning/launch_c51_fc320_eps3125.sh \
#     > /tmp/chain.log 2>&1 &
#
# Exists so a wave can be queued behind a close-out without anyone sitting on the terminal. The laptop
# has no job queue — that is the desktop's `ops` branch — so this is the minimum that does the job.
#
# **`chart_viewer` must be excluded from the poll, and this is the trap that makes the script worth
# having.** A laptop eval opens its own viewer via `chart_viewer.spawn_for_eval()`, whose command line
# contains `--watch eval_checkpoints.py <prefix>`. A naive `pgrep -f eval_checkpoints.py` matches it, the
# viewer outlives the evals by design, and the wait would then never end. Same class as the documented
# `pgrep snek2.py` matching git telemetry.
#
# **A zombie counts as alive to `kill -0` but not to a `pgrep` count**, which is why this polls the
# process list rather than tracking pids: a finished eval whose parent has not reaped it would otherwise
# hold the chain open forever.
set -u
cd "$(dirname "$0")/.." || exit 1

LAUNCHER=${1:?usage: chain_after_evals.sh <launcher-script> [poll-seconds]}
POLL=${2:-60}
[ -f "$LAUNCHER" ] || { echo "no such launcher: $LAUNCHER"; exit 1; }

echo "$(date '+%F %T')  waiting for eval_checkpoints.py to drain, polling every ${POLL}s"

while true; do
  n=$(pgrep -fl "eval_checkpoints.py" | grep python | grep -vc chart_viewer || true)
  n=${n:-0}
  if [ "$n" -eq 0 ]; then
    echo "$(date '+%F %T')  evals drained"
    break
  fi
  echo "$(date '+%F %T')  $n eval process(es) still running"
  sleep "$POLL"
done

# A close-out writes its results file with os.replace, so by the time the process is gone the JSON is
# complete. No settle time is needed for correctness; this is only to let the forked workers exit so the
# new wave starts on an idle box.
sleep 20

echo "$(date '+%F %T')  launching $LAUNCHER"
exec bash "$LAUNCHER"
