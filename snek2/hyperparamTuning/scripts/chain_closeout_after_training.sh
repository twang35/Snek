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
# **The HOF stage was added 2026-08-18 to close a host asymmetry, not to add a feature.** The desktop
# daemon has chained `training -> closeout -> HOF re-measure` off every training since 2026-08-15
# (`auto_hof`, default on), so a laptop batch measured with this script produced *less* than the same
# batch would have on the desktop, and the missing half was the one that decides whether a checkpoint
# is hall-of-fame material. The recipe below is copied from `desktop/runner/runner.py`'s
# `HOF_EVAL_ENV`/`HOF_EVAL_ARGS` so the two hosts produce comparable files; if that changes, change
# this too. The script's name is now slightly narrow — it was kept because it is referenced from
# several docs and a rename buys nothing.
#
# Three traps, all of them documented failures elsewhere in this project:
#
#   * `pgrep -f snek2.py` matches any command line *containing* the string — git pathspecs and the
#     Airbnb telemetry `curl` whose JSON payload names `snek2/snek2.py`. It once read 6 trainers when 4
#     were running. So a line counts only if it also runs `python`.
#   * `pgrep -f eval_checkpoints.py` matches `chart_viewer --watch eval_checkpoints.py <prefix>`, which
#     outlives the evals by design, so the eval drain must exclude it or it never ends.
#   * Both polls read the process list rather than tracking pids, because `kill -0` succeeds on a
#     zombie and forked self-eval workers briefly outlive their parent.
#
# Launching any eval displaces every chart at `evals/` top level into `evals/archive/<timestamp>/`.
# That is expected here — this *is* the batch whose charts should be current — but it is why the script
# refuses to start while someone else's close-out is still running.
set -u
cd "$(dirname "$0")/../.." || exit 1   # scripts/ -> hyperparamTuning/ -> snek2/, whatever the caller's cwd is

PREFIX=${1:?usage: chain_closeout_after_training.sh <batch-prefix> [poll-seconds]}
POLL=${2:-120}
WORKERS=${EVAL_WORKERS:-4}
COUNT=${CLOSEOUT_TOP:-top20}
LOGDIR=${CLOSEOUT_LOGDIR:-/tmp}
PY=${SNEK_PYTHON:-/opt/miniconda3/envs/snek/bin/python}

# Pinned rather than inherited, and pinned to the same values the desktop uses
# (runner.py CLOSEOUT_THRESHOLD / HOF_THRESHOLD). Two reasons the close-out gate must be set here:
# `eval_checkpoints` defaults to 95, so a laptop close-out and a desktop close-out of the same batch
# were being written under *different* gates — and CLAUDE.md is explicit that a file's gate lives in
# its payload as `min_achievable` and must be checked before anything is pooled across files.
CLOSEOUT_GATE=${CLOSEOUT_GATE:-96}
HOF_GATE=${HOF_GATE:-98}
CHAIN_HOF=${CHAIN_HOF:-1}

# The same invariant the desktop asserts. HOF selects `above:$HOF_GATE` *from the close-out's own
# result file*, and only rows that reach the close-out gate are measured full length — so a close-out
# gate at or above the HOF gate abandons the very rows the re-measure needs and silently starves it.
if [ "$CLOSEOUT_GATE" -ge "$HOF_GATE" ]; then
  echo "FAIL: CLOSEOUT_GATE=$CLOSEOUT_GATE must stay below HOF_GATE=$HOF_GATE, or the HOF pass has" >&2
  echo "      nothing full-length to select from." >&2
  exit 1
fi

trainers() { pgrep -fl "snek2.py ${PREFIX}" | grep python | grep -vc chart_viewer || true; }
evals()    { pgrep -fl "eval_checkpoints.py" | grep python | grep -vc chart_viewer || true; }

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

arms=$(ls -d savedPolicies/${PREFIX}[a-z]-* 2>/dev/null | sed 's|.*/||')
[ -n "$arms" ] || { echo "$(date '+%F %T')  no savedPolicies/${PREFIX}* arms found"; exit 1; }
n_arms=$(echo "$arms" | wc -l | tr -d ' ')

# ---------------------------------------------------------------- stage 1: the close-out
echo "$(date '+%F %T')  close-out: ${n_arms} arms x ${WORKERS} workers, ${COUNT}, gate ${CLOSEOUT_GATE}"
pids=""
for a in $arms; do
  echo "  -> $a  (log ${LOGDIR}/${a}_closeout.log)"
  EVAL_WORKERS="$WORKERS" EVAL_MIN_ACHIEVABLE="$CLOSEOUT_GATE" PYTHONPATH=. nohup "$PY" -u \
    eval_checkpoints.py "$a" "$COUNT" > "${LOGDIR}/${a}_closeout.log" 2>&1 &
  pids="$pids $!:$a"
done

# Per-arm exit codes, not a bare `wait`. An arm whose close-out died must not get a HOF pass: the
# HOF selector reads the close-out's result file, so a truncated file would quietly re-measure a
# partial selection and look like a finished, empty result.
closeout_ok=""
for pa in $pids; do
  p=${pa%%:*}; a=${pa##*:}
  if wait "$p"; then
    closeout_ok="$closeout_ok $a"
  else
    echo "$(date '+%F %T')  WARNING: close-out FAILED for $a — see ${LOGDIR}/${a}_closeout.log; no HOF for it"
  fi
done
echo "$(date '+%F %T')  all ${PREFIX} close-outs finished"

if [ "$CHAIN_HOF" != "1" ]; then
  echo "$(date '+%F %T')  CHAIN_HOF=$CHAIN_HOF — stopping before the HOF re-measure"
  exit 0
fi
[ -n "$closeout_ok" ] || { echo "$(date '+%F %T')  no close-out succeeded; nothing to re-measure"; exit 1; }

# ---------------------------------------------------------------- stage 2: the HOF-500 re-measure
# `complete: true` in the result file, not just a zero exit. The desktop gets this for free because
# it only sets its `hof: pending` marker once the close-out has been reaped; here the file itself is
# the authority on whether the selection it is about to be read for is finished.
hof_arms=""
for a in $closeout_ok; do
  "$PY" - "$a" "$HOF_GATE" <<'PYEOF'
import json, os, sys
policy, gate = sys.argv[1], float(sys.argv[2])
path = os.path.join('runs', policy + '_checkpoint_evals.json')
try:
    payload = json.load(open(path))
except Exception as e:
    print('   {0}: cannot read {1} ({2})'.format(policy, path, type(e).__name__)); sys.exit(1)
if not payload.get('complete'):
    print('   {0}: close-out file is NOT complete — skipping its HOF'.format(policy)); sys.exit(1)
above = [r for r in payload.get('results', [])
         if not r.get('abandoned') and r.get('perfect_percent', 0) >= gate]
print('   {0}: close-out complete (gate {1}), {2} checkpoint(s) >= {3:g}%'.format(
    policy, payload.get('min_achievable'), len(above), gate))
sys.exit(0 if above else 2)
PYEOF
  status=$?
  case "$status" in
    0) hof_arms="$hof_arms $a" ;;
    2) : ;;   # nothing cleared the gate. Normal for most arms, not a failure.
    *) echo "$(date '+%F %T')  WARNING: cannot select HOF checkpoints for $a (exit $status)" ;;
  esac
done

if [ -z "$hof_arms" ]; then
  echo "$(date '+%F %T')  no ${PREFIX} arm has a >=${HOF_GATE}% close-out checkpoint — no HOF pass owed."
  echo "$(date '+%F %T')  that is the normal outcome for most batches, not a failure. Done."
  exit 0
fi

echo "$(date '+%F %T')  HOF-500: $(echo $hof_arms | wc -w | tr -d ' ') arm(s) x ${WORKERS} workers," \
     "500 episodes flat, gate ${HOF_GATE}, suffix _hof500"
pids=""
for a in $hof_arms; do
  echo "  -> $a  (log ${LOGDIR}/${a}_hof500.log)"
  EVAL_WORKERS="$WORKERS" \
  EVAL_EPISODES=500 \
  EVAL_SCREEN_EPISODES=0 \
  EVAL_INDEPENDENT=1 \
  EVAL_MIN_ACHIEVABLE="$HOF_GATE" \
  EVAL_OUT_SUFFIX=_hof500 \
  PYTHONPATH=. nohup "$PY" -u \
    eval_checkpoints.py "$a" "above:${HOF_GATE}" > "${LOGDIR}/${a}_hof500.log" 2>&1 &
  pids="$pids $!:$a"
done

fails=0
for pa in $pids; do
  p=${pa%%:*}; a=${pa##*:}
  wait "$p" || { echo "$(date '+%F %T')  WARNING: HOF-500 FAILED for $a — ${LOGDIR}/${a}_hof500.log"; fails=$((fails+1)); }
done
echo "$(date '+%F %T')  all ${PREFIX} HOF-500 re-measures finished ($fails failed)"
echo "$(date '+%F %T')  results in runs/<arm>_checkpoint_evals_hof500.json"
echo "$(date '+%F %T')  NOTE: promotion into hallOfFame/ is still the manual, verified process."
exit 0
