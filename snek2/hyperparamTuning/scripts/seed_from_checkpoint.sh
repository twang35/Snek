#!/usr/bin/env bash
# Seeds a fresh policy dir from ONE checkpoint of an existing arm, so a training run resumes
# from that exact step instead of the source arm's latest.
#
# Why a fresh dir rather than resuming the source arm in place:
#   - common.Checkpointer.initialize_or_restore() restores whatever the `checkpoint` state file
#     names, which is the source's LAST step (2000000), not its best one.
#   - resuming in place would also append the continuation's checkpoints and graph history to the
#     original arm's record, i.e. overwrite the evidence the batch is being selected from.
# A dir holding exactly one pre-existing checkpoint means every later ckpt-* in it belongs to the
# new run, so its close-out eval cannot mix the two arms' weights at the same step.
#
# The replay buffer is carried across deliberately. A fresh dir has only the 1000 random
# transitions random_play() writes, and training samples a batch from those on step 1 -- a hard
# perturbation to a 98%-perfect policy. The source's buffer is from its 2M endpoint, so it is
# slightly ahead of the checkpoint being restored, but it is on-policy data from the same arm and
# is far closer than random play. PER re-prioritises against the restored net within a few
# thousand samples either way.
#
# Usage, from anywhere:
#   bash hyperparamTuning/scripts/seed_from_checkpoint.sh <src-policy> <step> <dst-policy> [savedPolicies-root]
#
# The root is last and optional so the common local case is three arguments, but it stays settable
# because this is also run on the desktop over ssh, where the repo is at a different absolute path:
#   scp hyperparamTuning/scripts/seed_from_checkpoint.sh the-claw-den:/tmp/
#   ssh the-claw-den 'bash /tmp/seed_from_checkpoint.sh b29b-... 1447000 b42a-... $HOME/Snek/snek2/savedPolicies'
# It is scp'd to /tmp rather than run from the box's checkout on purpose: an untracked file under
# snek2/ there is what aborts the deploy ff-merge (see desktop/README.md).
set -euo pipefail

src="$1"       # source policy name
step="$2"      # checkpoint step to resume from
dst="$3"       # new policy name
# Two hops: scripts/ -> hyperparamTuning/ -> snek2/. Same convention as the launchers here.
root="${4:-$(cd "$(dirname "$0")/../.." && pwd)/savedPolicies}"

s="$root/$src"
d="$root/$dst"

[ -d "$s" ] || { echo "FAIL: no source dir $s" >&2; exit 1; }
[ -f "$s/ckpt-$step.index" ] || { echo "FAIL: no ckpt-$step.index in $s" >&2; exit 1; }
[ -f "$s/ckpt-$step.data-00000-of-00001" ] || { echo "FAIL: no ckpt-$step data in $s" >&2; exit 1; }
[ -f "$s/arch.json" ] || { echo "FAIL: no arch.json in $s -- it will not load" >&2; exit 1; }
[ -e "$d" ] && { echo "FAIL: $d already exists; refusing to clobber" >&2; exit 1; }

mkdir -p "$d"
cp "$s/arch.json" "$d/arch.json"
cp "$s/ckpt-$step.index" "$s/ckpt-$step.data-00000-of-00001" "$d/"
if [ -f "$s/replay_buffer/buffer.npz" ]; then
  mkdir -p "$d/replay_buffer"
  cp "$s/replay_buffer/buffer.npz" "$d/replay_buffer/buffer.npz"
  buf=yes
else
  buf="NO (will random-populate)"
fi

# TF's CheckpointState proto, text format. Only this one entry, so the manager's rotation starts
# clean and does not believe it owns the source's other 1987 checkpoints.
printf 'model_checkpoint_path: "ckpt-%s"\nall_model_checkpoint_paths: "ckpt-%s"\n' "$step" "$step" \
  > "$d/checkpoint"

echo "seeded $dst  <- $src @ $step   replay_buffer=$buf"
ls -1 "$d" | sed 's/^/    /'
