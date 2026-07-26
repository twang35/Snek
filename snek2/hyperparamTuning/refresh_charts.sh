#!/bin/zsh
# Re-copies the live progress graphs from snek2/runs/ into charts/ and prints the
# step each one is at, so charts.md captions can be updated to match.
#
# The charts are deliberately copies rather than links: runs/ holds the live files
# that every eval rewrites, and if that directory were ever cleaned out the charts
# in the tuning docs would silently disappear.
set -u
HERE=${0:a:h}
RUNS=$HERE/../runs
PY=/opt/miniconda3/envs/snek/bin/python

mkdir -p $HERE/charts
for graph in $RUNS/*.png; do
  policy=${${graph:t}%.png}
  [[ $policy == smoke || $policy == tunetest ]] && continue
  cp $graph $HERE/charts/$policy.png
  step=$($PY - "$RUNS/${policy}_evals.json" <<'EOF' 2>/dev/null || echo "?"
import json, sys
rows = json.load(open(sys.argv[1]))['evals']
print(rows[-1]['step'])
EOF
)
  echo "$policy -> charts/$policy.png (step $step)"
done
