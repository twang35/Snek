#!/bin/bash
# Batch 39: **C51 initialised at expected Q = 0** instead of the grid midpoint, `eps 1e-4`, to 3M.
#
#   cd snek2
#   bash hyperparamTuning/scripts/launch_b39_zeroinit.sh
#
# **b36's config with `SNEK_C51_ZERO_INIT=1` as the only change** — same `eps 1.5e-4`, same `lr 1e-4`,
# same `fc 320`, same seeds 1-4, same 3M cap. So `b36a-d` is an exact seed-matched one-variable control and
# anything b39 does is attributable to the ramp. **The first arm in this project to run with it** — the knob
# shipped with `plans/distributional-c51.md` on 2026-08-15 and has been dead code until now, which is why
# the launch is preceded by a smoke check (see VERIFY at the bottom).
#
# **This is the first deliberate test of a mechanism the measurement says should LOSE.** The 2026-08-17
# reading (`perDiagnostics/init_optimism.py`) established that the true value on a shared champion state
# set is **~34**, so the standard init's 57.5 is 23.5 too high and **zero is 34.0 too low** — the ramp
# moves the init *further* from the truth, not closer. It is being run anyway because "it is worse" and
# "here is the mechanism by which it is worse" are different results, and only the second one transfers.
#
# **What the ramp actually does, measured rather than assumed.** `bias_i = -lambda*(z_i - v_min)` with
# **lambda = 0.16219** (bisected by `categorical_agent.zero_init_lambda`, not pasted):
#
#   standard init   E[Q] 58.52   aeff 49.87 of 51   bottom-3 mass 0.018
#   zero init       E[Q]  0.07   aeff  6.66 of 51   bottom-3 mass 0.290
#
# So one parameter sets the mean *and* the spread, and there is no way to separate them with a linear
# ramp. Two consequences, and the second is the one to watch:
#
#   1. The head starts **sharper than any trained net in this project ever becomes** (b36 settles at
#      aeff 20.9-24.6). Standard init starts at 49.9 and sharpens monotonically, which is the right
#      direction of travel for a distribution; this has to **broaden first, then sharpen**.
#   2. The ramp is a **-20.3 logit handicap on the top atom** (-0.16219 * 125) and **-17.0 at z=100**,
#      which is where a near-win has to put its mass. e^-17 is 4e-8, so high-return atoms start at
#      effectively zero probability and the head has to grow weights to climb out. At `lr 1e-4`, Adam
#      needs of order **200,000 steps** of consistently-signed gradient to move a bias by 20 logits.
#
# **Point 2 was confirmed on a 2,500-step smoke run before this batch was launched** (2026-08-17), which
# is worth recording because it is the mechanism rather than the outcome. Restoring the smoke checkpoint
# at step 2,000 and reading the head bias against the predicted ramp:
#
#   predicted   atom0 -0.000   atom42 (z=100) -17.030   atom50 (z=120) -20.273
#   actual      atom0 +0.009 to +0.013         -17.030                 -20.273
#
# **The top-atom biases had not moved at all to three decimals.** All the drift - max 0.0150 logits across
# all three actions - sat in the *bottom* atoms, where the early targets are (score 1.4, so every return
# is near zero). At that rate the bias alone would need ~2.7M steps to unwind 20 logits, so recovery has
# to come through the **kernel**, and it cannot start until the agent actually experiences high returns.
#
# **That is not a deadlock, and the distinction is the point.** Targets come from real rewards, so once the
# agent eats a lot, mass gets pushed upward and the atoms unsuppress. What is delayed is not the
# *discovery* of good play - exploration does that, at epsilon 0.4 with `guided_fraction 0.8` - but the
# *valuation* of it. Standard init is ready to represent a 100-return from step 0; this is not. So expect
# the damage, if any, in the endgame value signal rather than in whether the arm learns to play.
#
# One more reading from that smoke net: `E[Q]` on a zero observation came out `[-0.023, -0.078, -0.115]`,
# centred as designed, but the **action spread is ~0.09 against standard init's 0.73**. C51's one
# incidental advantage over `ddqn` here was fast action separation (gap 14.91 by 8k against the scalar
# control's 1.68 at 7k), so whether the ramp delays *that* is worth watching alongside the rest.
#
# **Adam's epsilon is held at b36's 1.5e-4 deliberately, and it interacts with the ramp.** Adam's step is
# `lr * mhat / (sqrt(vhat) + eps)`. The suppressed top atoms start at p ~ 4e-8, so their logit gradient
# `(p - target)` is tiny and `sqrt(vhat) << eps`, which makes the effective step on exactly those
# parameters `lr * mhat / eps` — **damped in proportion to `eps`**. So a larger Adam epsilon slows the
# recovery of the atoms zero-init suppressed, and this batch is not run at the friendliest value for the
# ramp. **That is the right trade**: matching b36 exactly buys clean attribution, which is worth more than
# giving the treatment its best shot, and the damping is a property of the treatment rather than a
# handicap imposed on it. Anyone re-running this at a smaller `eps` should expect the ramp to unwind
# faster, and should not read that as the ramp working better.
#
# **Pre-registered, in order of expected likelihood.** Written before launch so the post-hoc story cannot
# drift; full reasoning in `runs.md`.
#
#   H1 (~50%)  Null on best-30 and `sef` (inside b36/b38's 80.0-88.3), with a **measurably slower first
#              100-200k steps**. The offset is common-mode so it cannot corrupt argmax, and 3M is far
#              longer than the combined calibration burden. Tell: time-to-first-80%-eval up, endpoint same.
#   H2 (~35%)  A real regression of 5-10 pp, from the top-atom suppression delaying the *endgame* value
#              signal specifically. Tell: `value_by_length.py` flat across bands 85-97 early, and
#              `endgame_gradient.py` showing a smaller endgame action gap than b36's 19.8-24.3.
#   H3 (~15%)  An improvement, via less early over-optimistic bootstrapping and therefore less churn.
#              Tell: `c51_stability.py --states-from` below b36's 0.0865 at a matched `--end`.
#
# H2 was raised from 30% and H1 lowered when the epsilon was set to 1.5e-4 rather than 1e-4, for the
# damping reason above: the ramp unwinds through parameters whose Adam step is divided by `eps`.
#
# **The measurement that separates all three is `aeff` against step**, and it is a signature no other arm
# can produce: standard init falls 49.9 -> 21-24 monotonically, so if b39 climbs 6.7 -> ~36 -> 21-24 the
# non-monotonic path is direct evidence the head spent training broadening before it could sharpen. Run:
#
#   PYTHONPATH=. python hyperparamTuning/perDiagnostics/init_optimism.py \
#     --policy b39a-c51zeroinitseed1 --policy b36a-c51fc320seed1 \
#     --states-from hallOfFame/b29b-chase10g75seed2-ckpt1447000 --states 1500 --points 15
#
# Judge on: that `aeff` path, then best-30 and `sef` against b36 at a matched 2M, then churn.
set -u
cd "$(dirname "$0")/../.." || exit 1   # scripts/ -> hyperparamTuning/ -> snek2/, whatever the caller's cwd is

PY=/opt/miniconda3/envs/snek/bin/python
STEPS=${STEPS:-3000000}

running=$(pgrep -fl "python -u snek2.py" | grep -c python)
if [ "$running" -gt 0 ]; then
  echo "ABORT: $running trainer(s) already running; this wave is 4 and the laptop limit is 4."
  exit 1
fi

# A close-out saturates the box, and this wave would slow it for hours. `chart_viewer` is excluded because
# a laptop eval spawns one whose argv contains `--watch eval_checkpoints.py <prefix>` and which outlives
# the evals — the same trap `chain_after_evals.sh` documents.
evals=$(pgrep -fl "eval_checkpoints.py" | grep python | grep -vc chart_viewer || true)
if [ "${evals:-0}" -gt 0 ]; then
  echo "ABORT: $evals eval process(es) still running; wait for the close-out to finish."
  exit 1
fi

letter_for() { case $1 in 1) echo a;; 2) echo b;; 3) echo c;; 4) echo d;; esac; }

for seed in 1 2 3 4; do
  name="b39$(letter_for $seed)-c51zeroinitseed${seed}"
  SNEK_ALGO=c51 SNEK_NUM_ATOMS=51 SNEK_V_MIN=-5 SNEK_V_MAX=120 \
    SNEK_C51_ZERO_INIT=1 \
    SNEK_FC_LAYERS=320 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
    SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
    SNEK_LEARNING_RATE=1e-4 SNEK_ADAM_EPSILON=1.5e-4 \
    SNEK_SEED="$seed" SNEK_MAX_STEPS="$STEPS" \
    PYTHONPATH=. "$PY" -u snek2.py "$name" > "/tmp/$name.log" 2>&1 &
  echo "  $name  fc 320  zero-init  eps 1.5e-4  seed $seed  pid $!"
  # Staggered, so the viewer's claim lock and arm registry are not what is being tested.
  sleep 5
done

echo "4 arms launched to ${STEPS} steps; one shared chart window on --arms b39"
echo "VERIFY: grep 'zero-expected-Q' /tmp/b39a-c51zeroinitseed1.log  -- the knob is silent if it fails"
