#!/bin/bash
# Batch 32: does Adam's `epsilon` separate C51's learning speed from its churn?
#
#   cd snek2
#   bash hyperparamTuning/launch_adam_eps.sh
#
# Four arms to 1M, `lr 1e-4` throughout, two seeds at each of the two published reference values:
#
#   b32a/b   eps 1.5e-4    (Dopamine's Rainbow config)
#   b32c/d   eps 3.125e-4  (Dopamine's C51 config)
#
# **The control is not in this wave — it is already on disk.** `c51pilotB-lr1e4seed1` and `seed2` ran
# this exact config at the Keras default `eps=1e-7` to 600k, so seeds 1 and 2 are reused here on
# purpose and the comparison is paired at a 600k horizon with the remaining 400k as free information.
# Reusing the seeds is not cosmetic: the seed decides which arm in a wave wins in 18 of 18 measured
# waves, and a paired design is the only thing that differences that out.
#
# `lr 1e-4` rather than the pilot's chosen `5e-5` because that is where the defect is largest while the
# rate still learns — churn 0.117-0.245 against the ddqn control's 0.033-0.058, never settling, yet
# seed 2 reached best-30 66.3 and was still rising at 599k. At `2.5e-4` the arm is broken outright and
# a working fix could be invisible underneath whatever else has gone wrong.
#
# **Judge this on churn and drawdown depth, not `best_perfect30`.** The within-rate seed spread at
# `1e-4` is 54.6 pp, so at n=2 per side the score resolves nothing. `c51_stability.py` is the readout:
#
#   PYTHONPATH=. python hyperparamTuning/perDiagnostics/c51_stability.py \
#     --policy b32a-c51eps15e4seed1 --policy b32c-c51eps3125e4seed1 \
#     --policy c51pilotB-lr1e4seed1 --end 600000
#
# The `b32` prefix means the trainers open **one** shared chart window between them, the normal path —
# no hand-started viewer here, unlike the `c51pilot*` waves whose names `batch_prefix` cannot group.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/opt/miniconda3/envs/snek/bin/python
STEPS=${STEPS:-1000000}

running=$(pgrep -fl "python -u snek2.py" | grep -c python)
if [ "$running" -gt 0 ]; then
  echo "ABORT: $running trainer(s) already running; this wave is 4 and the laptop limit is 4."
  exit 1
fi

launch() {
  name=$1; eps=$2; seed=$3
  SNEK_ALGO=c51 SNEK_NUM_ATOMS=51 SNEK_V_MIN=-5 SNEK_V_MAX=120 \
    SNEK_FC_LAYERS=200,100,100 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
    SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
    SNEK_LEARNING_RATE=1e-4 SNEK_ADAM_EPSILON="$eps" \
    SNEK_SEED="$seed" SNEK_MAX_STEPS="$STEPS" \
    PYTHONPATH=. "$PY" -u snek2.py "$name" > "/tmp/$name.log" 2>&1 &
  echo "  $name  eps $eps  seed $seed  pid $!"
  # Staggered, so the viewer's claim lock and arm registry are not the thing being tested tonight.
  sleep 5
}

launch b32a-c51eps15e4seed1   1.5e-4   1
launch b32b-c51eps15e4seed2   1.5e-4   2
launch b32c-c51eps3125e4seed1 3.125e-4 1
launch b32d-c51eps3125e4seed2 3.125e-4 2

echo "4 arms launched to ${STEPS} steps; one shared chart window on --arms b32"
