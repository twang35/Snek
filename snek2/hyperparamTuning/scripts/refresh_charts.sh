#!/bin/zsh
# Re-copies the live progress graphs from snek2/runs/ into charts/ and prints the
# step each one is at, so charts.md captions can be updated to match.
#
# The charts are deliberately copies rather than links: runs/ holds the live files
# that every eval rewrites, and if that directory were ever cleaned out the charts
# in the tuning docs would silently disappear.
set -u
# Every path is derived from the script's own location (`${0:a:h}`) rather than from the caller's
# cwd, so it works from anywhere and moving the file only needs the two `..` counts below changed.
HERE=${0:a:h}                 # snek2/hyperparamTuning/scripts
TUNING=$HERE/..               # snek2/hyperparamTuning — where charts/ lives
RUNS=$HERE/../../runs         # snek2/runs — where the live graphs are written
PY=/opt/miniconda3/envs/snek/bin/python

mkdir -p $TUNING/charts
for graph in $RUNS/*.png; do
  policy=${${graph:t}%.png}
  [[ $policy == smoke || $policy == tunetest ]] && continue
  # runs/ used to also hold <policy>_eval_progress.png, written by eval_progress.py /
  # eval_checkpoints.py. Those are close-out progress views, not an arm's training graph, and
  # copying them in made charts/ look like it held twice as many arms as it does — each one
  # showed up as an undocumented arm in the charts.md completeness check. They now write to
  # snek2/evals/ instead, so this exclusion only matters for files from before that change.
  [[ $policy == *_eval_progress ]] && continue
  cp $graph $TUNING/charts/$policy.png
  step=$($PY - "$RUNS/${policy}_evals.json" <<'EOF' 2>/dev/null || echo "?"
import json, sys
rows = json.load(open(sys.argv[1]))['evals']
print(rows[-1]['step'])
EOF
)
  echo "$policy -> charts/$policy.png (step $step)"
done
