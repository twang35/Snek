#!/usr/bin/env bash
# b43 -- the laptop half of the "keep training a champion" experiment.
#
# Same four checkpoints as b42 on the desktop, same b29 config, ONE change:
# SNEK_LEARNING_RATE=1e-6 instead of the default 1e-5. b42 is the seed-matched control.
#
# This batch only measures anything because of training.enforce_learning_rate (2026-08-18).
# Adam's `learning_rate` is a checkpointed tf.Variable, so initialize_or_restore() silently
# restored the 1e-5 these checkpoints were trained at and SNEK_LEARNING_RATE was a no-op on
# every resume. Each arm prints
#   learning rate: checkpoint restored 1e-05, reset to the configured 1e-06
# at startup -- if that line is missing, the arm is training at 1e-5 and the batch is a null.
#
# The policy dirs are pre-seeded (arch.json + one ckpt pair + the source arm's replay_buffer +
# a `checkpoint` state file naming only that step), so each arm resumes from its source's BEST
# checkpoint rather than the source's 2M endpoint. 3M is absolute, so an arm seeded at 1.447M
# trains ~1.55M more steps.
#
# Honest note on this file as a record: the four arms were launched at 20:52 from a version without
# the trainer-limit guard below (the laptop was checked by hand first). The guard was added when this
# was filed here, so re-running it refuses instead of making eight trainers. Everything the arms
# actually ran with -- the env block, the policy names, the 3M cap -- is verbatim.
set -u
# Two hops: scripts/ -> hyperparamTuning/ -> snek2/. The convention every script here follows, so
# the caller's cwd is never load-bearing. See ./README.md.
cd "$(dirname "$0")/../.." || exit 1

# Never break the 4-trainer laptop limit. `grep python` is load-bearing: `pgrep -f snek2.py` is a
# substring match and also catches git pathspecs and the Airbnb telemetry curl whose payload names
# snek2/snek2.py -- that once read 6 trainers against 4 running.
running=$(pgrep -fl "python -u snek2.py" 2>/dev/null | grep -c python || true)
if [ "${running:-0}" -gt 0 ]; then
  echo "refusing to launch: $running trainer(s) already running on this laptop" >&2
  pgrep -fl "python -u snek2.py" | grep python >&2
  exit 1
fi

PY=/opt/miniconda3/envs/snek/bin/python
LOGDIR=/tmp/b43-logs
mkdir -p "$LOGDIR"

launch () {  # $1 policy  $2 seed
  SNEK_LEARNING_RATE=1e-6 \
  SNEK_FC_LAYERS=320 \
  SNEK_IS_WEIGHTS=0 \
  SNEK_TARGET_UPDATE_PERIOD=1000 \
  SNEK_DISCOUNT=0.9975 \
  SNEK_FOOD_DISTANCE_REWARD=0 \
  SNEK_FORK_BRANCHES=4 \
  SNEK_SEED="$2" \
  SNEK_CHASE_SAFE_SHAPING=0.1 \
  SNEK_CHASE_SAFE_GATE=75 \
  SNEK_MAX_STEPS=3000000 \
  PYTHONPATH=. nohup "$PY" -u snek2.py "$1" > "$LOGDIR/$1.log" 2>&1 &
  echo "launched $1 (seed $2) pid $!"
}

launch b43a-lowlr-b29b 2
launch b43b-lowlr-b29a 1
launch b43c-lowlr-b40b 2
launch b43d-lowlr-b29c 3
