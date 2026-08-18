#!/bin/bash
# Wait for a batch's trainers to drain, then run its close-out as 4 parallel eval processes.
#
#   cd snek2
#   nohup bash hyperparamTuning/chain_closeout_after_training.sh b39 > /tmp/b39_closeout_chain.log 2>&1 &
#
# The mirror of `chain_after_evals.sh`: that one queues a *wave* behind a close-out, this one queues a
# *close-out* behind a wave. Arms self-terminate at `SNEK_MAX_STEPS`, so nobody has to sit on the
# terminal waiting to launch the measurement.
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
cd "$(dirname "$0")/.." || exit 1

PREFIX=${1:?usage: chain_closeout_after_training.sh <batch-prefix> [poll-seconds]}
POLL=${2:-120}
WORKERS=${EVAL_WORKERS:-4}
COUNT=${CLOSEOUT_TOP:-top20}
LOGDIR=${CLOSEOUT_LOGDIR:-/tmp}

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

echo "$(date '+%F %T')  close-out: $(echo "$arms" | wc -l | tr -d ' ') arms x ${WORKERS} workers, ${COUNT}"
for a in $arms; do
  echo "  -> $a  (log ${LOGDIR}/${a}_closeout.log)"
  EVAL_WORKERS="$WORKERS" PYTHONPATH=. nohup /opt/miniconda3/envs/snek/bin/python -u \
    eval_checkpoints.py "$a" "$COUNT" > "${LOGDIR}/${a}_closeout.log" 2>&1 &
done

wait
echo "$(date '+%F %T')  all ${PREFIX} close-outs finished"
