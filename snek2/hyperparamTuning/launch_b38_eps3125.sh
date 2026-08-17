#!/bin/bash
# Batch 38: batch 36 at the **other** Adam epsilon — `3.125e-4` instead of `1.5e-4`, to 3M.
#
#   cd snek2
#   bash hyperparamTuning/launch_b38_eps3125.sh
#
# Identical to [`launch_c51_fc320.sh`](launch_c51_fc320.sh) — `ALGO=c51`, 51 atoms over `[-5, 120]`,
# `fc 320`, `lr 1e-4`, `IS_WEIGHTS=0`, `TARGET_UPDATE_PERIOD=1000`, `DISCOUNT=0.9975`,
# `FORK_BRANCHES=4`, no food-distance shaping, win reward at its default 100, seeds 1-4 — with
# **`SNEK_ADAM_EPSILON=3.125e-4`** the only change. So `b36a-d` is an exact seed-matched control and
# this is a clean one-variable dose comparison on the new architecture.
#
# **This is the dose question b32 could not answer, retried with 4 seeds a side instead of 2.** On b32's
# `fc 200,100,100` the two epsilon values were indistinguishable on the shared-state-set churn reading —
# `1.5e-4` **0.0865** against `3.125e-4` **0.0895** — but that was n=2 per side, which was stated before
# launch as unable to resolve a 2x dose. b36 + b38 gives **4 seeds per side on one architecture**, which is
# the first configuration in this project able to say anything about the dose at all.
#
# **Read it against b36 on churn first, with `--states-from`.** A cross-arm churn comparison is only
# legitimate on a shared state set: the per-arm sets inflated b32's epsilon effect ~2x, because a weak arm
# dies early and is then scored on near-tied early-game states that flip for free. Use the same reference
# every other C51 reading now uses, so the numbers are comparable across batches:
#
#   PYTHONPATH=. python hyperparamTuning/perDiagnostics/c51_stability.py \
#     --policy b36a-c51fc320seed1 ... --policy b38a-c51fc320eps3125seed1 ... \
#     --states 1500 --stride 5000 --points 10 --end 2000000 \
#     --states-from hallOfFame/b29b-chase10g75seed2-ckpt1447000
#
# **What each outcome means.** Lower churn at `3.125e-4` with best-30 held would make the higher dose the
# C51 default and suggest the response is still climbing at 3.125e-4, worth one more rung. Lower churn
# *with* worse best-30 is epsilon acting as a smaller learning rate in disguise — the known failure mode,
# and the reason best-30 is read alongside rather than after. A dead heat at n=4 a side would close the
# dose question for good and leave `1.5e-4` as the default on the grounds that it is the lower-variance
# reference config.
#
# b36 stopped at 1.87-2.02M rather than its 3M cap, so **match at 2M** when comparing, not at the caps.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/opt/miniconda3/envs/snek/bin/python
STEPS=${STEPS:-3000000}

running=$(pgrep -fl "python -u snek2.py" | grep -c python)
if [ "$running" -gt 0 ]; then
  echo "ABORT: $running trainer(s) already running; this wave is 4 and the laptop limit is 4."
  exit 1
fi

# Evals saturate the box, and a wave launched on top of one gets a third of the cores. The chaining
# script waits these out; this is the backstop if the launcher is run by hand too early. Excludes
# chart_viewer, whose `--watch eval_checkpoints.py <prefix>` argv matches a naive grep.
evals=$(pgrep -fl "eval_checkpoints.py" | grep python | grep -vc chart_viewer || true)
if [ "${evals:-0}" -gt 0 ]; then
  echo "ABORT: $evals eval process(es) still running; wait for the close-out to finish."
  exit 1
fi

letter_for() { case $1 in 1) echo a;; 2) echo b;; 3) echo c;; 4) echo d;; esac; }

for seed in 1 2 3 4; do
  name="b38$(letter_for $seed)-c51fc320eps3125seed${seed}"
  SNEK_ALGO=c51 SNEK_NUM_ATOMS=51 SNEK_V_MIN=-5 SNEK_V_MAX=120 \
    SNEK_FC_LAYERS=320 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
    SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
    SNEK_LEARNING_RATE=1e-4 SNEK_ADAM_EPSILON=3.125e-4 \
    SNEK_SEED="$seed" SNEK_MAX_STEPS="$STEPS" \
    PYTHONPATH=. "$PY" -u snek2.py "$name" > "/tmp/$name.log" 2>&1 &
  echo "  $name  fc 320  eps 3.125e-4  seed $seed  pid $!"
  # Staggered, so the viewer's claim lock and arm registry are not what is being tested.
  sleep 5
done

echo "4 arms launched to ${STEPS} steps; one shared chart window on --arms b38"
