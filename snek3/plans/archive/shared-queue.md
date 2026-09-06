# One queue for both boxes — design proposal (2026-09-05)

Status: **built 2026-09-06** (decision 7: rsync). `tools/claims.py`, `tools/batch_state.py`, the scheduler's
`--shared`, the daemon without its mirror, `progress_update` over both feeds, the `box` column on the page,
`queue-batch` in place of `desktop-batch` and `move-batch`. Section 8 is what the build changed from the plan.
Section 0 is the
problem, 1 is the mechanism and why it beats the alternatives, 2 is the design piece by piece, 3 is what
changes in each file, 4 the migration, 5 the phasing, 6 the decisions, 7 what happens with one box or a
late start.

## 0. The problem

The two boxes each run their own queue, and a person moves work between them (`move-batch`). They finish at
different times — tonight the desktop clears ~11:00 and the laptop ~00:30 the day after — so one sits idle
while the other has a day of work, and every rebalance is a hand-run skill with a priority trap in it
(2026-09-05 22:18: b18 arrived at 200 beside b20's 220 and a restarted scheduler picked it over the live wave).

Wanted: **one queue, both boxes pull from it**, a spec can be pinned to a box or left for either, and the
two boxes cannot pick the same work when they both come free at once. Same code on both boxes, as always.

## 1. The mechanism: claim by pushing, and let git be the lock

A `git push` of a fast-forward commit is an **atomic compare-and-swap on a ref**: the server accepts it
only if the branch is still at the commit the push was built on. That is exactly the primitive a shared
queue needs, and the bus already gives every box a git remote from anywhere.

**A box claims a wave by committing a small file to a `claims` branch and pushing.** If two boxes race —
both fetched `claims` at commit S, both committed a claim on top of S — the server takes one push and
rejects the other as non-fast-forward. The loser resets to the new head, re-reads the claims (its wave is
now someone's), recomputes the next free wave, commits and pushes again. Two boxes idle at the same instant
claim two *different* waves within a second or two, with no coordinator and no second phase.

| | the daemon assigns (the idea in the question) | **claim by push** |
|---|---|---|
| phases | three: the box asks (a field in its status), the daemon assigns on a network cycle, the box reads the assignment on its next poll | one: the box pushes its claim; the push succeeding *is* the assignment |
| latency | up to `git_seconds` (600 s) plus the laptop's own cycle, or an ssh trigger | one fetch and one push |
| depends on | the desktop daemon being alive and on-bus; the laptop cannot get work while the box is down | the git remote only — the same dependency the bus already has |
| where the logic lives | in the daemon, the one component that is deliberately stdlib-only and knows nothing about waves; the laptop would carry none of it | in `tools/`, once, run by both schedulers |
| the race | moved, not removed: the daemon must still decide whether a *previous* assignment is stale, and the asking box must not double-ask | settled by the server, which is why it needs no timeout, no lease and no tie-break |

Two other shapes considered and dropped. **A static split** (the desktop takes waves from the front of the
order, the laptop from the back) needs no coordination but fails whenever one box is off, and is the
imbalance we have now. **Claiming on `ops` itself** (rewriting a spec's `box` field) makes `ops` a
multi-writer branch shared with a person, which is the failure the bus was shaped to avoid; the claims
branch keeps `ops` person-written and read-only for the machines.

**About the one-writer-per-branch rule.** It exists because two *feeds* on one branch need a merge, and a
merge is a failure with no owner. The `claims` branch is not a feed: it is a lock, the loser of a race
discards its commit and recomputes rather than merging, and the file it writes is a few hundred bytes.
So the rule stands, with `claims` as the one deliberate exception, written down as such in
`desktop/README.md`'s branch table.

## 2. The design

### 2.1 What is claimed: a wave and its chain

A wave's stage B, hof5000 and hof30k read the arms' checkpoints from the box's own `savedPolicies/`, so a
wave and its passes are one unit and stay on one box. Finer than a wave gains nothing (both boxes run 8
wide; a partial wave wastes cores). Coarser — a batch — is today's problem: a 24-arm batch is ~10-12 h on
one box while the other idles. **A batch may therefore span both boxes, wave by wave.**

A claim is `claims/<batch>/w<N>.json`:

```json
{"batch": "b21", "wave": 3, "box": "laptop", "arms": ["b21ar-gate60-seed1", "..."],
 "claimed_iso": "2026-09-06T04:12:07", "claimed_ts": 1788671527.1}
```

| rule | why |
|---|---|
| **N = the batch's highest existing claim + 1**, whatever box made it | pass ids stay `b21-stageb`, `-w2`, `-w3` and are unique across both feeds and the site; `pass_label` is unchanged |
| the arms of the wave are **the next `wave_size` unclaimed train specs of the batch in `spec_order`** | the same grouping `waves(specs, size)` makes today when one box owns the batch, so a batch on one box runs exactly as now |
| a claim is made **just before the box would launch the wave**, never ahead | a claim held by a box that then sleeps is the one failure this design has (2.5); claiming late keeps that window to the wave being run |
| an **eval spec** is claimed whole, and only by a box that has every one of its policies under `savedPolicies/` | its checkpoints are wherever they were trained; a box probes its own disk, no branching on which box it is |
| a spec with `"box": "desktop"` or `"box": "laptop"` is claimable only by that box; absent means either | the pin the question asked for. The box's own name is `SNEK_BOX`, set by the daemon from `host.env` and defaulting to `laptop` — a name, not a code path |
| order across the pool: by batch (lowest `priority` of its specs, then name), within a batch by `priority` then id | the scheduler's existing order, now over one queue instead of two; the move-batch priority trap is gone because there is one order |

### 2.2 Where the claiming lives: the scheduler, on both boxes

`tools/claims.py`, new, ~250 lines, imported by the scheduler and callable by hand:

| function | does |
|---|---|
| `read(repo)` | fetch `ops` and `claims`; return every pending spec (via `job.parse_job` + `launch.materialise`, which the scheduler may import — `desktop/` shells out and imports nothing from `tools/`, and the dependency stays one-way) and every claim |
| `next_claim(specs, claims, box, wave_size, has_checkpoints)` | **pure**: the wave or eval spec this box should claim next, or None. Where every rule of 2.1 lives, and where the tests are |
| `claim(repo, record)` | commit the file in the `claims` worktree (`gitbus.ensure_worktree`, beside the status and results ones) and push. On a non-fast-forward rejection: reset to `origin/claims`, return "lost"; the caller re-reads and recomputes, up to a handful of tries |
| `release(repo, batch, wave)` | delete the claim file, same push; the wave's arms return to the pool and the *next* claimer numbers a new wave (`N+1`; a released number is never reused, so no pass id ever means two waves) |
| `mine(claims, box)` | the waves and eval specs this box holds |
| `show()` | the pool as the status wants it: unclaimed by batch, and each box's holdings. `python -m tools.claims show --json` |

**The scheduler's local queue directory becomes a mirror of this box's claims**, on both boxes. Today the
desktop daemon mirrors `ops` into `desktop/queue-local/<batch>/` and the laptop's specs are copied by hand
into `logs/laptop-queue/<batch>/`. Instead the scheduler itself, at every boundary, writes
`<queue>/<batch>/<id>.json` for every spec of every wave it holds — each spec carrying `"_wave": N` from
its claim — and removes what it no longer holds. `Driver` groups arms by `_wave` when the key is present
and chunks by size when it is not, so `tools.scheduler <specs>` by hand and every existing test keep
working; `finished`, `pending`, `run_wave`, `run_pass`, the registry, the window and the results feed are
untouched.

The loop (`run_queue` → `run_shared`):

```
loop:
    sync the mirror from my claims
    run every batch directory with pending work (the existing Driver, waves from `_wave`)
    when nothing I hold is pending:
        record = next_claim(...)            # re-reads ops and claims first
        if record is None: exit            # the queue's rule: it lives exactly as long as there is work
        claim(record); on "lost": continue  # someone took it; the next pass of the loop recomputes
```

So the desktop daemon's `_mirror_specs`, `_batches_with_work`, the `published` set that fed it, and the
`.done-<id>` marking all go: **the daemon no longer decides what the box runs.** With them goes the trap
the `desktop-batch` skill warns about — a hand eval on a closed batch waking the whole batch and retraining
arms whose `_evals.json` is not on the box — because nothing is mirrored that was not explicitly claimed.

### 2.3 The daemon: spawn on movement, publish the pool

The daemon keeps the bus, the actions, the runtime knobs, the status publish and the site build. Its
"is there work" check becomes: **spawn a scheduler when none is alive and the `ops` or `claims` head has
moved since the last spawn, or on a trigger**, under the existing 10-minute backoff. A scheduler that
finds nothing to claim exits in seconds, so the daemon needs no notion of a wave and imports nothing new;
this is the existing queue-signature rule with the signature now `(ops head, claims head)`.

`status.json` gains `at_a_glance.pool`: the unclaimed work by batch, with the pin if any (`b21 training |
24 arms unclaimed | ~9h`, `b7 eval | b7-hof30k-confirm | pinned laptop`). The daemon gets it by shelling
out to `tools.claims show --json` as it does for `site_build`, so the listing has one implementation.
`running`/`queued` stay the desktop's own, `laptop_*` the laptop's. `_track_finished` and the derived
`ledger` block go (2.6 replaces their one reader).

### 2.4 The laptop: the same scheduler, started when the laptop should join

`tools.scheduler --shared` on the laptop fetches `ops` and `claims` itself; `logs/laptop-queue/` is its
mirror. Lifetime stays as it is: **it exits when it can claim nothing**, so nothing runs while there is no
work, and **the laptop takes part exactly while its scheduler is up**. A laptop whose scheduler is down
claims nothing and the desktop takes the pool; starting the scheduler is how the laptop joins. The
queueing skill's last step is "start the laptop scheduler if none is up", so a batch pushed with the
laptop meant to help starts there within the minute. (The alternative — a laptop scheduler that idles
polling `ops` — is a second daemon; decision 6.2.)

### 2.5 The one failure: a claim whose box went quiet

A box claims a wave, then the laptop's lid closes for a day, or its scheduler is killed and not restarted.
The wave is neither running nor claimable. **No lease and no automatic expiry**: an expiry that fires
while the box is merely slow is two boxes training the same arms, which is worse than an idle wave.
Instead:

| | |
|---|---|
| detect | the daemon already folds `laptop_iso` in. New `attention` line when a box holds a claim whose arms are not on any feed and that box's status is older than two hours: `laptop holds b21-w3 (8 arms); laptop status 5.2h old` |
| release | `python -m tools.claims release b21-w3` — a human decision, in the `stop-run` skill: stop the wave on its box first if it is alive, then release. The arms go back to the pool and the next claimer takes them as a fresh wave; whatever was trained on the quiet box is abandoned (a box cannot resume another's checkpoints, as today) |
| guard on the quiet box | the scheduler re-reads claims at each boundary; an arm live on this box whose wave it no longer holds is logged loudly and left alone (a running trainer is never killed by a sync), and its files are never published — it will not reach the feed because the scheduler does not `_settle` an arm it does not hold |

### 2.6 Results, the site, the progress update

| piece | today | change |
|---|---|---|
| results feeds | one per box, single writer, the scheduler publishes at three moments | **none.** A batch split across boxes puts wave 1 on `results` and wave 2 on `laptop-results`, each under the ids the claims numbered |
| `site_build` | flattens both feeds into one directory by file name | **none** for correctness; add `box` to the manifest record from the claims, so a split batch's page says which arms ran where (the two boxes' `steps_per_second` differ and would otherwise read as an effect) |
| `progress_update.batch_state` | counts states from the desktop's derived `ledger`; `laptop_state_line` reads the laptop's `runs/` separately | one `tools/batch_state.py` for both: a batch's arms from `ops`; each arm's box and wave from the claims, done from the feeds (`results/<id>/` or `laptop-results/<id>/` exists), running from whichever box's status lists it. One line per wave: `b21: w1 desktop closed, w2 laptop training 40%, w3 unclaimed` |
| `wave_close_times`, `import_closed_waves` | read `results` only | both feeds |
| `at_a_glance` | desktop `running`/`queued`, laptop `laptop_*` | plus `pool` (2.3) |

### 2.7 Skills

| skill | becomes |
|---|---|
| `desktop-batch` | **`queue-batch`**: write the specs (with `box` when pinned), validate with `parse_job`, name the reference cell, push `ops`, trigger, and start the laptop scheduler if the laptop should join. The approval rule is unchanged: pushing to `ops` starts real work |
| `move-batch` | retired. Its two cases become one line each in `queue-batch` (pin: edit `box` on the unclaimed specs and push) and `stop-run` (release: `tools.claims release`) |
| `laptop-run` | the queue section becomes "start `tools.scheduler --shared` if none is up"; smoke and one-off unchanged |
| `stop-run` | after stopping a wave, whether to release its claim, and how |
| `progress-update` | procedure unchanged; the tool changes underneath |

## 3. What changes, by file

| file | change | size |
|---|---|---|
| `tools/claims.py` | new: read, `next_claim`, claim/release with the fast-forward retry, `show` | ~250 |
| `tools/scheduler.py` | `--shared`; mirror from claims; `_wave` grouping; claim at the boundary; `pool` and holdings in the status; the not-mine guard | +150 / −60 (`queue_batches` stays for the bare form) |
| `tools/laptop_status.py` | `box` from `SNEK_BOX` | ±5 |
| `desktop/daemon/daemon.py` | delete `_mirror_specs`, `_batches_with_work`, `_track_finished`, `_finished`, `_seed_published_from_ledger`, the derived `ledger`; spawn on `(ops, claims)` head movement; `pool` from `tools.claims show --json` | −250 |
| `desktop/daemon/launch.py` | `--shared` in `scheduler_command`; `SNEK_BOX=desktop` in `scheduler_env`; `materialise` stays here (the scheduler imports it) | ±15 |
| `desktop/daemon/gitbus.py` | a plain (non-force) `push` variant that reports "rejected" distinctly from "failed" | +25 |
| `desktop/daemon/job.py` | optional `box` field, validated against `{desktop, laptop}` | +10 |
| `tools/batch_state.py` | new: a batch's per-wave state from ops + claims + feeds + both statuses; `progress_update` and `viewer_manifest` read it | ~120 |
| `tools/progress_update.py` | use `batch_state`; import and close times from both feeds; delete `laptop_state_line` | ±100 |
| `tools/viewer_manifest.py` | `box` per arm | +15 |
| `tests/` | `next_claim` cases (pin, eligibility, numbering, order); the race against a local bare `origin` with two worktrees pushing on the same parent; the mirror; the daemon's spawn rule; `batch_state` | ~400 |
| skills, `desktop/README.md`, `CLAUDE.md`, `plans/archive/scheduler.md` | 2.7, the branch table, the one-writer exception, the two-hosts section | docs |

## 4. Migration

Live at the time of writing: the desktop has b17 with two waves left and b21 (24 arms) queued on `ops`;
the laptop has b20 wave 1 training, b20 wave 2 and b18 (24 arms) in `logs/laptop-queue/`.

1. **Seed `claims` from what has run**, once, by hand (`tools.claims seed`, a one-off that reads both
   boxes' `runs/`): every wave already trained or training becomes a claim with its box and its number,
   so pass ids continue where they are (`b17-stageb-w7` is still wave 7). Batches closed before today
   need no claims — nothing will look for them.
2. **Put b18's specs back on `ops`**, and every laptop-queue batch that has not started. Set `priority`
   deliberately across the pool: it now orders one queue, so b21 at 220 and b18 at 240 means b21's
   waves go first to whichever box comes free.
3. Deploy at a **wave boundary on each box** (the tidy moment; a restart is safe at any point, but a
   scheduler started mid-wave must find its wave in the seeded claims, and step 1 guarantees that only
   if it ran after the wave started). Laptop first, as with the scheduler itself: the shared mode is
   developed in a worktree, since the laptop's arms run from the working tree.
4. Watch the first boundary on each box claim the right wave, then delete `move-batch`.

## 5. Phasing

| phase | lands | verified by |
|---|---|---|
| 1 | `tools/claims.py` and its tests, including the race | two worktrees push on the same parent against a local bare remote; exactly one wins, the other recomputes and claims the next wave |
| 2 | scheduler `--shared` on the laptop; `job.py`'s `box`; `laptop_status`'s box name | a two-arm smoke batch on `ops` pinned `laptop`, claimed and run end to end; then b18 through the real pool |
| 3 | daemon: delete the mirror, spawn on movement, `pool` in the status; deploy | the desktop claims b21's next wave at its boundary; `status.json` shows the pool once and each box's holdings |
| 4 | `batch_state`, `progress_update`, manifest `box` | a split batch's table and state line read right |
| 5 | skills and docs; retire `move-batch` | the next queueing goes through `queue-batch` |

## 6. Decisions (user, 2026-09-05)

| # | question | decision |
|---|---|---|
| 1 | claim unit: wave, or whole batch | **wave.** A batch spanning boxes costs only a `box` column on the page; the balance is the point |
| 2 | laptop scheduler lifetime | **exits when idle, as now.** The laptop joins by starting it; a scheduler that polls forever is a second daemon with a second set of liveness rules |
| 3 | stranded claims | **manual release with a loud `attention` line, no expiry.** An expiry that misfires trains the same arms twice |
| 4 | claim timing | **just before launch**, never one ahead: the smallest possible stranded window, and the pool in the status is what is really free |
| 5 | pin syntax | `"box": "desktop" \| "laptop"`, absent for either. No `any` keyword to misspell |
| 6 | the pass numbering across boxes | **global per batch from the claims**, so `b21-stageb-w2` means one wave everywhere; a released number is never reused |
| 7 | a hand eval spec over arms whose checkpoints are not all on one box | **open** -- see below |
| 8 | retire `move-batch` | **yes**, and delete its content from every other doc and tool that carries it (`CLAUDE.md`'s two-hosts table, `desktop/README.md`, `laptop-run`, `desktop-batch`, `hyperparam-sweep.md`'s move notes) rather than marking it obsolete |

**Decision 7, spelled out.** The chain's passes are per wave and never hit this: a wave's arms and their
checkpoints are on the box that trained them. The case is a **hand eval spec** -- `b7-hof30k-confirm`, a
`hof-remeasure`, a `one` re-measure -- whose `policies` list arms whose `savedPolicies/<arm>/` are not all on
one box. Today that is a batch trained before the desktop existed (checkpoints on the laptop only, so the
laptop claims it, no problem) or one whose checkpoints were pruned. **Under this plan it also becomes every
batch that was split across the boxes**: an eval spec over all 24 arms of a split b21 is claimable by
neither box, because each holds 16 of the checkpoints. Two ways to handle it:

| | a. rsync, as now | b. the spec runs in parts |
|---|---|---|
| what happens | the spec sits in the pool as `b21-hof30k-confirm | needs checkpoints: 8 arms not on this box`, from each box's view; a person rsyncs the missing `savedPolicies/` to one box (the `hof-remeasure` skill's existing step) and the spec is claimed on the next boundary | each box claims the spec **for the policies it holds** (`claims/b21/eval-b21-hof30k-confirm-laptop.json`), runs the close-out over those, publishes their files; the spec is done when every policy has its file on some feed |
| cost | a hand step per split-batch re-measure, and the checkpoints copied twice on disk | ~60 lines in `next_claim` and the mirror, one more claim shape, and a "done" test that reads both feeds |
| recommended | **for now.** Hand evals are rare (four so far) and the pool line says exactly what to copy | if split-batch re-measures become routine |

## 7. One box, or a late start

Both work, and nothing in the design assumes two boxes are up.

| situation | what happens |
|---|---|
| only the desktop is running | its scheduler claims the next free wave at every boundary until the pool holds nothing it may take. Specs pinned `laptop` stay in the pool, listed as such |
| only the laptop is running | the same. The desktop's daemon being down or off-bus changes nothing: the laptop reads `ops` and pushes `claims` itself |
| a box starts at any time | its scheduler reads `ops` and `claims`, first runs whatever it already holds (a wave it was mid-way through -- checkpoints local, arms resume; the just-committed "live arms first" rule, `2116545a1`, is the same behaviour), then claims the next free wave. There is no registration, no roster of boxes, and no state about a box anywhere but the claims it holds |
| a box goes away mid-wave | the wave stays claimed (2.5). It resumes when the box returns, or a person releases it |
| the desktop daemon starts a scheduler | on any movement of the `ops` or `claims` head, or a trigger, when none is alive (2.3). So the laptop claiming a wave wakes the desktop's scheduler too, which is right: the pool changed and the desktop may now be the one with work to take |
| the laptop scheduler | started by hand or by the `queue-batch` skill; exits when it can claim nothing (decision 2). A dead laptop scheduler is simply a box that claims nothing |

"Unclaimed" is purely *not in any claim*: a finished wave is claimed, a running wave is claimed, and the
claimant never needs to know whether the other box exists.

## 8. As built (2026-09-06)

| planned | built |
|---|---|
| `read(repo)` fetching and parsing | `read_specs` off the fetched `ops` ref, in **two git calls** for the whole directory (`gitbus.read_pending_jobs`: `ls-tree` + `cat-file --batch`); a `git show` per spec was ~a minute on the laptop, whose git carries a telemetry wrapper, and the scheduler reads this at every boundary |
| the daemon shells out to `tools.claims show --json` for the pool | as planned, once per network cycle; the view also carries a **derived `ledger`** (done from the feeds, running from both statuses) because `tools/viewer_manifest.py` reads one for the page's pass states -- the block the earlier plan dropped is kept as a derived view, one implementation, in `claims.ledger` |
| `batch_state.py` for the progress update | as planned; `progress_update` also imports finished jobs from **both** feeds (a job on a feed is finished by construction) and writes `runs/.live/boxes.json`, which `site_build` writes on the desktop from which feed carried the arm, for the page's `box` tag |
| the mirror rewritten by the scheduler | `claims.mirror`, called from `SharedQueue.sync` at the top of every pass of `run_shared`; markers are kept, a spec no longer on `ops` is logged and not mirrored, and an arm live here that no claim covers is an `attention` line (`SharedQueue.unheld`), never killed |
| a `gitbus` push variant | `push_fast_forward`: `'pushed'`, `'rejected'` (lost the race) or `'failed'`; on either non-success the store resets the worktree to the remote so it never drifts |
| the migration's seed | `tools.claims seed <batch> <box> <waves>`, cutting waves as the scheduler did (`spec_order`, eight at a time); and `ops` pruned of every closed batch -- ~400 specs, each of which the pool would have offered |
