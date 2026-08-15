#!/bin/bash
# Phase 3 of plans/distributional-c51.md: the C51 pilot screen, four arms.
#
# `{lr 1e-5, lr 5e-5} x {seed 1, seed 2}` on b25's config verbatim plus the c51 knobs, capped at
# 600k steps. Seed-matched across the two rates, so the pair differs in one thing. A screen, not a
# result: does it learn at all, how fast to the first perfect game, is the loss scale sane at 1e-5.
#
# It **waits for b30's close-out to finish first**, because the plan's host note says the pilot
# should not share 14 cores with four eval processes. Nothing is launched until the last
# `eval_checkpoints.py b30` process is gone.
#
#   cd snek2 && nohup bash hyperparamTuning/launch_c51_pilot.sh > /tmp/c51pilot-launcher.log 2>&1 &
#
# `SNEK_CHART_VIEWER=0` on every arm, and one viewer opened by hand: `chart_viewer.batch_prefix`
# only groups `b<n><letters>-` names, so four `c51pilot-*` arms would each open a window of their
# own. The `--glob`/`--watch` form is the same one an eval wave uses.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/opt/miniconda3/envs/snek/bin/python

# `pgrep -f` matches any command line *containing* the pattern, and the eval wave's own chart viewer
# carries `--watch eval_checkpoints.py b30` in its argv — so without the two filters this loop waits
# on the viewer forever. Same trap as `pgrep -fl watch` matching `watchman` (CLAUDE.md).
running_evals() {
  pgrep -fl "eval_checkpoints.py b30" | grep -v chart_viewer | grep -c python
}

while [ "$(running_evals)" -gt 0 ]; do
  echo "$(date '+%H:%M:%S') waiting for b30's close-out ($(running_evals) processes)"
  sleep 60
done
echo "$(date '+%H:%M:%S') close-out done — launching the c51 pilot"

launch() {
  local name=$1 lr=$2 seed=$3
  SNEK_ALGO=c51 SNEK_NUM_ATOMS=51 SNEK_V_MIN=-5 SNEK_V_MAX=120 \
    SNEK_FC_LAYERS=200,100,100 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
    SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
    SNEK_LEARNING_RATE="$lr" SNEK_SEED="$seed" SNEK_MAX_STEPS=600000 \
    SNEK_CHART_VIEWER=0 \
    PYTHONPATH=. "$PY" -u snek2.py "$name" > "/tmp/$name.log" 2>&1 &
  echo "  $name  lr $lr  seed $seed  pid $!"
}

launch c51pilot-lr1e5seed1 1e-5 1
launch c51pilot-lr1e5seed2 1e-5 2
launch c51pilot-lr5e5seed1 5e-5 1
launch c51pilot-lr5e5seed2 5e-5 2

# The viewer needs at least one PNG to show, and the first one lands with the first eval. Wait for it
# rather than sleeping a guessed interval: an empty glob is a window with nothing in it.
for _ in $(seq 1 40); do
  if compgen -G 'runs/c51pilot-*.png' > /dev/null; then break; fi
  sleep 15
done
"$PY" -u chart_viewer.py --glob 'runs/c51pilot-*.png' \
  --watch 'snek2.py c51pilot' --title 'c51 pilot — live' > /tmp/c51pilot-viewer.log 2>&1 &
echo "  viewer pid $!"

wait
echo "$(date '+%H:%M:%S') all four arms exited"
