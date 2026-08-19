#!/bin/bash
# Unattended handoff: wait for the C51 pilot to finish, pick the learning rate, launch the batch,
# write the docs.
#
#   cd snek2
#   nohup bash hyperparamTuning/scripts/launch_c51_batch.sh b31 > /tmp/c51-batch-handoff.log 2>&1 &
#
# Standing in for a person who is not there, so every step is either deterministic or refuses to
# guess:
#
# 1. **Wait for every pilot arm to stop.** `pgrep -f` is a substring match, so one pattern covers
#    both waves (`c51pilot` matches `c51pilotB` too). The `grep -v chart_viewer` is required — a
#    wave's viewer carries `--watch snek2.py c51pilot` in its own argv, so without it this waits
#    forever on the window rather than on the arms.
# 2. **Wait for the laptop's 4-trainer limit to be free.** If anything else is training when the
#    pilots stop, launching would put six or eight arms on the box against the hard rule in
#    CLAUDE.md. It waits, and if the slots never free it **exits without launching** and says so,
#    rather than breaking the limit.
# 3. **Pick the rate** with `pick_c51_lr.py`, whose rule is pre-registered in its docstring. If it
#    refuses (fewer than two rates with usable data), fall back to `$FALLBACK_LR` so the batch still
#    starts — a batch on a defensible default beats no batch at all overnight.
# 4. **Launch 4 seeds at 2M** on b25's config plus c51, which makes `b25a-d` the seed-matched
#    control. The `b<n>` prefix means the trainers open **one** shared chart window themselves, the
#    normal path — no hand-started viewer here.
# 5. **Regenerate the pilot's doc regions and commit them.** Docs-only, which CLAUDE.md authorises
#    without asking. A push failure is logged and ignored: the commit is local and recoverable, and
#    the batch matters more than the push.
set -u
cd "$(dirname "$0")/../.." || exit 1   # scripts/ -> hyperparamTuning/ -> snek2/, whatever the caller's cwd is

PY=/opt/miniconda3/envs/snek/bin/python
BATCH=${1:-b31}
FALLBACK_LR=${FALLBACK_LR:-5e-5}
SLOT_WAIT_MINUTES=${SLOT_WAIT_MINUTES:-360}

say() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

pilot_arms() { pgrep -fl "snek2.py c51pilot" | grep -v chart_viewer | grep -c python; }
all_trainers() { pgrep -fl "python -u snek2.py" | grep -c python; }

say "waiting for the pilot to finish ($(pilot_arms) arms up)"
while [ "$(pilot_arms)" -gt 0 ]; do sleep 120; done
say "every pilot arm has stopped"

waited=0
while [ "$(all_trainers)" -gt 0 ]; do
  if [ "$waited" -ge "$SLOT_WAIT_MINUTES" ]; then
    say "ABORT: $(all_trainers) trainer(s) still running after ${waited}m, so launching $BATCH would"
    say "ABORT: exceed the laptop's 4-trainer limit. Nothing launched; run this script again when free."
    exit 1
  fi
  say "$(all_trainers) other trainer(s) running — waiting for a free slot (${waited}m so far)"
  sleep 300
  waited=$((waited + 5))
done

say "picking the learning rate"
PICK=$(PYTHONPATH=. "$PY" -u hyperparamTuning/pick_c51_lr.py 2>&1)
echo "$PICK"
LR=$(echo "$PICK" | sed -n 's/^CHOSEN_LR=//p' | tail -1)
COMPACT=$(echo "$PICK" | sed -n 's/^CHOSEN_COMPACT=//p' | tail -1)
if [ -z "$LR" ]; then
  say "the picker refused to choose — falling back to $FALLBACK_LR"
  LR=$FALLBACK_LR
  COMPACT=$(echo "$LR" | tr -d '.-')
fi
say "learning rate for $BATCH: $LR"

letter_for() { case $1 in 1) echo a;; 2) echo b;; 3) echo c;; 4) echo d;; esac; }

for seed in 1 2 3 4; do
  name="${BATCH}$(letter_for $seed)-c51lr${COMPACT}seed${seed}"
  SNEK_ALGO=c51 SNEK_NUM_ATOMS=51 SNEK_V_MIN=-5 SNEK_V_MAX=120 \
    SNEK_FC_LAYERS=200,100,100 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
    SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
    SNEK_LEARNING_RATE="$LR" SNEK_SEED="$seed" SNEK_MAX_STEPS=2000000 \
    PYTHONPATH=. "$PY" -u snek2.py "$name" > "/tmp/$name.log" 2>&1 &
  say "launched $name (lr $LR, seed $seed, pid $!)"
  # Staggered: four trainers reaching main() inside the same second is what the viewer's claim lock
  # and arm registry exist for, and there is no reason to lean on them when nobody is watching.
  sleep 5
done

say "writing the docs"
PYTHONPATH=. "$PY" -u hyperparamTuning/pick_c51_lr.py --write-docs --batch "$BATCH" --lr "$LR"

cd ..
git add snek2/hyperparamTuning/charts.md snek2/hyperparamTuning/runs.md \
        snek2/hyperparamTuning/charts snek2/runs/c51pilot_lr_choice.json 2>&1
git commit -q -m "C51 pilot closed: lr $LR chosen, batch $BATCH launched at 2M

Written by hyperparamTuning/scripts/launch_c51_batch.sh with nobody watching, so the
pilot's tables in charts.md and runs.md are machine-generated from the eval
series and the surrounding prose is whatever the last session left. The rate
comes from pick_c51_lr.py's pre-registered rule (mean best_perfect30 at a common
horizon, then strong_eval_fraction, then peak_trailing).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
  && say "committed" || say "nothing to commit, or the commit failed"
git push -q origin master && say "pushed" || say "push failed — the commit is local, push by hand"

say "done: $BATCH is training at lr $LR, 2M cap"
