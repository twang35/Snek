#!/bin/bash
# Batch 33: what happens when a filled board pays 10 instead of 100?
#
#   cd snek2
#   bash hyperparamTuning/scripts/launch_win10.sh
#
# Four arms to **3M**, `SNEK_PERFECT_GAME_REWARD=10`, `SNEK_V_MAX=40`, otherwise batch 32's config at
# `eps 1.5e-4` — so `b32a`/`b32b` (same `eps`, same `lr`, seeds 1 and 2) are an **exact paired control
# differing only in the win reward**, and seeds 3-4 add spread. Deliberately *not* a mirror of b32's
# 2+2 epsilon split: this batch varies the reward, so everything else is held at one setting.
#
# **`v_max=40` is measured, not 120/10.** The return from a state is `F + gamma^(T-t) * W`, and `F` —
# the remaining food discounted to now — is largest at the *start* of an episode. At `W=100` the win
# dominates and the maximum return is just before winning (104.4). At `W=10` it does not, so the
# maximum moves to the opening: `return_distribution.py` on `b18b-ckpt1588000` with the reward changed
# measures **32.46** (16 episodes, seeds 301+302, gamma 0.9975), and 40 gives 21% headroom — the same
# proportion the shipped 120 has over 104.4. Spacing 0.9, so `FOOD_REWARD` is **1.11 atoms** against
# 0.40 today, which is the resolution gain this change buys.
#
# **This is expected to underperform, and the point is to see the shape of the failure.** Two concrete
# predictions, both from measurements already in findings.md:
#
# 1. **The value ordering over states inverts.** At `W=10` a length-20 state is worth ~19 and a
#    length-98 state ~11, because 95 discounted meals beat four meals plus a 10-point win. At `W=100`
#    the endgame was the high-value region. So the agent gets no value gradient pulling it toward
#    finishing.
# 2. **Urgency to finish drops 10x.** Delaying the win 100 steps costs `W*(1 - 0.9975^100)`: 22 reward
#    at 100, 2.2 at 10. Endgame hunting speed is the measured elite-vs-mediocre discriminator (p90
#    steps/meal 5-13 for the records against 86-226) and starvation is the modal failure at median
#    length 98, so watch **steps per meal at length 85+** and the starve/death split, not just best-30.
#
# If it fails, `behaviour_profile.py` and `point_of_no_return.py` are the scripts that say *how* —
# both read a c51 arm unchanged.
set -u
cd "$(dirname "$0")/../.." || exit 1   # scripts/ -> hyperparamTuning/ -> snek2/, whatever the caller's cwd is

PY=/opt/miniconda3/envs/snek/bin/python
STEPS=${STEPS:-3000000}

# 8 trainers is over the standing 4-arm rule and is the user's explicit call for this pair of batches,
# the same suspension the C51 pilot ran under. Measured then: ~2.3 GB per arm against 36 GB of RAM, so
# the cost is throughput rather than paging. Refuse anything beyond that.
running=$(pgrep -fl "python -u snek2.py" | grep -c python)
if [ "$running" -gt 4 ]; then
  echo "ABORT: $running trainers already running; this wave is 4 and 8 total is the authorised ceiling."
  exit 1
fi

letter_for() { case $1 in 1) echo a;; 2) echo b;; 3) echo c;; 4) echo d;; esac; }

for seed in 1 2 3 4; do
  name="b33$(letter_for $seed)-c51win10seed${seed}"
  SNEK_ALGO=c51 SNEK_NUM_ATOMS=51 SNEK_V_MIN=-5 SNEK_V_MAX=40 \
    SNEK_PERFECT_GAME_REWARD=10 \
    SNEK_FC_LAYERS=200,100,100 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
    SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
    SNEK_LEARNING_RATE=1e-4 SNEK_ADAM_EPSILON=1.5e-4 \
    SNEK_SEED="$seed" SNEK_MAX_STEPS="$STEPS" \
    PYTHONPATH=. "$PY" -u snek2.py "$name" > "/tmp/$name.log" 2>&1 &
  echo "  $name  win 10  seed $seed  pid $!"
  sleep 5
done

echo "4 arms launched to ${STEPS} steps; own chart window on --arms b33"
