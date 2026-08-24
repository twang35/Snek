#!/bin/bash
# Wait for a batch's trainers to drain, run its close-out, then run its HOF-500 re-measure.
#
#   cd snek2
#   nohup bash hyperparamTuning/scripts/chain_closeout_after_training.sh b43 > /tmp/b43_chain.log 2>&1 &
#
# The mirror of `chain_after_evals.sh`: that one queues a *wave* behind a close-out, this one queues a
# *close-out* behind a wave. Arms self-terminate at `SNEK_MAX_STEPS`, so nobody has to sit on the
# terminal waiting to launch the measurement.
#
# **This script's only job now is the wait.** It used to carry the whole two-stage measurement: four
# `eval_checkpoints.py` processes, per-arm pid bookkeeping, an inline python check that each close-out
# came out `complete`, a hand-copied HOF recipe (500 episodes, flat, `_hof500`) and its own copy of the
# `closeout gate < HOF gate` invariant. Every one of those was a second copy of something
# `eval_plan.hof_settings` and `--chain` now own, and the copies were the failure mode: the header used
# to say "copied from desktop/runner/runner.py; if that changes, change this too". So the stages are one
# `--chain` call, and this file is a drain-poll plus a launch.
#
# **Which engine.** `SNEK_EVAL_ENGINE=vec` (the default) runs `vectorized/vec_wave.py`;
# `SNEK_EVAL_ENGINE=scalar` runs `eval_wave.py`. The vectorised engine measures ~40x faster and was
# validated against the TF path at four levels, ending in a 24-checkpoint x 500-episode head-to-head
# that agreed to -0.058 pp (z = -0.28). It refuses c51 policies and hands them to `eval_wave.py`
# itself, so a categorical batch needs no opt-out.
#
# Two traps, both documented failures elsewhere in this project:
#
#   * `pgrep -f snek2.py` matches any command line *containing* the string — git pathspecs and the
#     Airbnb telemetry `curl` whose JSON payload names `snek2/snek2.py`. It once read 6 trainers when 4
#     were running. So a line counts only if it also runs `python`.
#   * `pgrep -f <eval tool>` matches `chart_viewer --watch <eval tool> <prefix>`, which outlives the
#     evals by design, so the eval drain must exclude it or it never ends.
#
# Both polls read the process list rather than tracking pids, because `kill -0` succeeds on a zombie
# and forked self-eval workers briefly outlive their parent.
#
# Launching any eval displaces every chart at `evals/` top level into `evals/archive/<timestamp>/`.
# That is expected here — this *is* the batch whose charts should be current — but it is why the script
# refuses to start while someone else's close-out is still running.
set -u
cd "$(dirname "$0")/../.." || exit 1   # scripts/ -> hyperparamTuning/ -> snek2/, whatever the caller's cwd is

PREFIX=${1:?usage: chain_closeout_after_training.sh <batch-prefix> [poll-seconds]}
POLL=${2:-120}
COUNT=${CLOSEOUT_TOP:-top50}
LOGDIR=${CLOSEOUT_LOGDIR:-/tmp}
PY=${SNEK_PYTHON:-/opt/miniconda3/envs/snek/bin/python}
ENGINE=${SNEK_EVAL_ENGINE:-vec}
CHAIN_HOF=${CHAIN_HOF:-1}

case "$ENGINE" in
  vec)    TOOL=vectorized/vec_wave.py ;;
  scalar) TOOL=eval_wave.py ;;
  *) echo "FAIL: SNEK_EVAL_ENGINE=$ENGINE — expected 'vec' or 'scalar'" >&2; exit 1 ;;
esac

# Deliberately *not* set here, for either engine. The gates, the episode counts, the screen depth and
# the `_hof500` suffix are `eval_plan.py`'s, and this script carrying values for them is exactly how a
# laptop close-out and a desktop close-out of the same batch ended up written under different gates.
# `EVAL_WORKERS`/`EVAL_LANES` (scalar) and `VEC_WAVE_PROCS` (vec) are inherited if the caller sets them,
# since they size the box rather than the protocol.
trainers() { pgrep -fl "snek2.py ${PREFIX}" | grep python | grep -vc chart_viewer || true; }
# Every eval entry point, not just the one this script launches: any of them contends for the
# cores and has its charts displaced by ours. `pgrep -f` takes an ERE, so this is one pattern.
EVAL_TOOLS="eval_checkpoints.py|eval_wave.py|vec_eval.py|vec_wave.py"
evals()    { pgrep -fl "$EVAL_TOOLS" | grep python | grep -vc chart_viewer || true; }

echo "$(date '+%F %T')  waiting for ${PREFIX} trainers to drain, polling every ${POLL}s"
while true; do
  n=$(trainers); n=${n:-0}
  [ "$n" -eq 0 ] && { echo "$(date '+%F %T')  ${PREFIX} trainers drained"; break; }
  echo "$(date '+%F %T')  $n ${PREFIX} trainer(s) still running"
  sleep "$POLL"
done

# Someone else's close-out would both contend for the cores and have its charts displaced by ours.
while true; do
  n=$(evals); n=${n:-0}
  [ "$n" -eq 0 ] && break
  echo "$(date '+%F %T')  waiting: $n other eval process(es) running"
  sleep "$POLL"
done

sleep 20   # let the trainers' forked self-eval workers exit so the close-out starts on an idle box

# The batch prefix, not an expanded arm list: both waves resolve `b43` to its arms themselves
# (`eval_wave.resolve_policies`), which is what retired this script's `ls -d savedPolicies/<prefix>[a-z]-*`
# glob and what lets the desktop's job spec carry a batch id.
CHAIN=--chain
[ "$CHAIN_HOF" = "1" ] || CHAIN=""
LOG="${LOGDIR}/${PREFIX}_closeout.log"

echo "$(date '+%F %T')  ${ENGINE} engine: $TOOL ${CHAIN:+--chain }${COUNT} ${PREFIX}"
echo "$(date '+%F %T')  log ${LOG}"
[ -n "$CHAIN" ] || echo "$(date '+%F %T')  CHAIN_HOF=$CHAIN_HOF — close-out only, no HOF re-measure"

# Foreground, and the exit code is the wave's. One process owns the whole batch now, so there is no
# per-arm pid bookkeeping left to do and no way for a dead arm to be handed a HOF pass out of a
# truncated file — `--chain` reads `complete` from each arm's own result before selecting from it.
PYTHONPATH=. "$PY" -u "$TOOL" $CHAIN "$COUNT" "$PREFIX" > "$LOG" 2>&1
status=$?

echo "$(date '+%F %T')  ${PREFIX} measurement finished (exit ${status})"
echo "$(date '+%F %T')  results in runs/<arm>_checkpoint_evals.json and _checkpoint_evals_hof500.json"
echo "$(date '+%F %T')  NOTE: promotion into hallOfFame/ is still the manual, verified process."
exit $status
