#!/bin/bash
# Batch 36: C51 on **fc 320** — one wide layer instead of three narrow ones, to 3M.
#
#   cd snek2
#   bash hyperparamTuning/scripts/launch_c51_fc320.sh
#
# Batch 32's config **verbatim at `eps 1.5e-4`** — `ALGO=c51`, 51 atoms over `[-5, 120]`, `lr 1e-4`,
# `IS_WEIGHTS=0`, `TARGET_UPDATE_PERIOD=1000`, `DISCOUNT=0.9975`, `FORK_BRANCHES=4`, no food-distance
# shaping, win reward back at its default **100** — with `SNEK_FC_LAYERS=320` the only change. Seeds 1-4,
# **3M cap**.
#
# **Two controls, both already on disk, and they answer different questions.**
#
#   b32a/b32b   same eps, same lr, same seeds 1-2, `fc 200,100,100` — the *architecture* pair. Clean:
#               one variable. But only 1M deep, so match at 1M before quoting anything.
#   b24a-d      `fc 320`, IS off, same seeds — *ddqn* at this exact shape, 2M, closed out, pooled 87.9
#               with two ≥98%/500 records. The "is C51 worth it at all" comparison.
#
# **3M is the point as much as the shape.** No C51 arm has ever run past **1M** — the pilot stopped at
# 600k, b31 at ~560k, b32 at its 1M cap — so the horizon is untested for this algorithm. b32's best-30
# peaks landed at 353-865k and every b33 arm declined for 1.4M steps after peaking, so "does C51 hold or
# decay past 1M" is a real question with a real cost attached: if it decays, every future C51 batch can
# stop at ~1.2M and cost a third as much.
#
# **What the shape actually changes for a categorical head.** Parameters, obs 30 -> 3x51 = 153 outputs:
#
#   fc 320          30*320+320 = 9,920   then 320*153+153 = 48,960   total ~58.9k, **83% in the last layer**
#   fc 200,100,100  6,200 + 20,100 + 10,100 + 15,453                 total ~51.9k, **30% in the last layer**
#
# So this is not mainly "more capacity" (+13%) — it moves the parameter budget into the layer that feeds
# the 153-way distribution, and removes two layers for gradient noise to compound through. The second half
# is why churn is the reading to watch: C51's defect here is instability, not level.
#
# **The pre-registered expectation is a null on the ceiling.** `findings.md` has nine shapes and
# architecture never raising it, and the one direct measurement says the deeper net's penultimate layer was
# **not** capacity-bound (effective rank 16-20 of 100, head outputs 4-6 of 153). If that rank comes out at
# 16-20 of 320 here too, widening bought nothing and any gain is optimisation, not capacity — check it
# rather than assuming either way.
#
# Judge on: churn at a **matched** horizon (`c51_stability.py --end 1000000`), best-30 against b32's 70.0
# group mean at 1M, then the ≥98%/500 count at close-out against b24's two.
set -u
cd "$(dirname "$0")/../.." || exit 1   # scripts/ -> hyperparamTuning/ -> snek2/, whatever the caller's cwd is

PY=/opt/miniconda3/envs/snek/bin/python
STEPS=${STEPS:-3000000}

running=$(pgrep -fl "python -u snek2.py" | grep -c python)
if [ "$running" -gt 0 ]; then
  echo "ABORT: $running trainer(s) already running; this wave is 4 and the laptop limit is 4."
  exit 1
fi

letter_for() { case $1 in 1) echo a;; 2) echo b;; 3) echo c;; 4) echo d;; esac; }

for seed in 1 2 3 4; do
  name="b36$(letter_for $seed)-c51fc320seed${seed}"
  SNEK_ALGO=c51 SNEK_NUM_ATOMS=51 SNEK_V_MIN=-5 SNEK_V_MAX=120 \
    SNEK_FC_LAYERS=320 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
    SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
    SNEK_LEARNING_RATE=1e-4 SNEK_ADAM_EPSILON=1.5e-4 \
    SNEK_SEED="$seed" SNEK_MAX_STEPS="$STEPS" \
    PYTHONPATH=. "$PY" -u snek2.py "$name" > "/tmp/$name.log" 2>&1 &
  echo "  $name  fc 320  eps 1.5e-4  seed $seed  pid $!"
  # Staggered, so the viewer's claim lock and arm registry are not what is being tested.
  sleep 5
done

echo "4 arms launched to ${STEPS} steps; one shared chart window on --arms b36"
