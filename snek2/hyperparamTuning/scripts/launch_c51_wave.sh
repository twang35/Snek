#!/bin/bash
# Launch a wave of C51 arms on b25's config, one process per arm, plus one chart window for the wave.
#
#   cd snek2
#   bash hyperparamTuning/scripts/launch_c51_wave.sh <prefix> <lr>:<seed> [<lr>:<seed> ...]
#
# e.g. the second pilot wave (2026-08-15):
#   bash hyperparamTuning/scripts/launch_c51_wave.sh c51pilotB 1e-4:1 1e-4:2 2.5e-4:1 2.5e-4:2
#
# The arm name is `<prefix>-lr<compact>seed<n>`, where `<compact>` drops the `-` and the `.`
# (`2.5e-4` -> `25e4`), so a name is a legal policy directory and still readable.
#
# **The wave-A launcher (`launch_c51_pilot.sh`) is kept as the record of that launch and must not be
# edited while it runs** — bash reads a script incrementally, so editing a running one can make it
# execute garbage. This file is the generic form; new waves come through here.
#
# `SNEK_CHART_VIEWER=0` on every arm and one viewer opened by hand, because
# `chart_viewer.batch_prefix` only groups `b<n><letters>-` names and each arm would otherwise open a
# window of its own. The `--watch` pattern is the *prefix*, so a wave's window closes when that
# wave's last arm stops. Note `pgrep -f` is a substring match, so a longer prefix that starts with a
# shorter one (`c51pilotB` inside `c51pilot`) keeps the shorter wave's window alive too — harmless,
# and it means the earlier wave's finished curves stay on screen as the reference.
set -u
cd "$(dirname "$0")/../.." || exit 1   # scripts/ -> hyperparamTuning/ -> snek2/, whatever the caller's cwd is

PY=/opt/miniconda3/envs/snek/bin/python

if [ "$#" -lt 2 ]; then
  sed -n '2,20p' "$0"
  exit 1
fi

PREFIX=$1
shift

for spec in "$@"; do
  lr=${spec%%:*}
  seed=${spec##*:}
  compact=$(echo "$lr" | tr -d '.-')
  name="$PREFIX-lr${compact}seed${seed}"
  SNEK_ALGO=c51 SNEK_NUM_ATOMS=51 SNEK_V_MIN=-5 SNEK_V_MAX=120 \
    SNEK_FC_LAYERS=200,100,100 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
    SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
    SNEK_LEARNING_RATE="$lr" SNEK_SEED="$seed" SNEK_MAX_STEPS=600000 \
    SNEK_CHART_VIEWER=0 \
    PYTHONPATH=. "$PY" -u snek2.py "$name" > "/tmp/$name.log" 2>&1 &
  echo "  $name  lr $lr  seed $seed  pid $!"
done

# The viewer needs at least one PNG, and the first lands with the first eval. Wait for it rather than
# sleeping a guessed interval: an empty glob is a window with nothing in it.
for _ in $(seq 1 40); do
  if compgen -G "runs/$PREFIX-*.png" > /dev/null; then break; fi
  sleep 15
done
"$PY" -u chart_viewer.py --glob "runs/$PREFIX-*.png" \
  --watch "snek2.py $PREFIX" --title "$PREFIX — live" > "/tmp/$PREFIX-viewer.log" 2>&1 &
echo "  viewer pid $!"
