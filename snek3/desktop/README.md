# The desktop — `the-claw-den`

**Not built yet.** This is a phase-4 deliverable
([`../plans/pytorch-port.md`](../plans/pytorch-port.md) §9). snek2's daemon still owns the box; the
working system and its incident log are at `../../snek2/desktop/README.md`, and the port is a
simplification of it rather than a redesign.

## What it will be

A stdlib-only systemd daemon on a dedicated Linux box that **imports nothing from this project** — it
shells out — and talks to the laptop only through three single-writer git branches. That decoupling is
the design's best property and is why the port keeps the architecture intact.

| branch | writer | payload |
|---|---|---|
| `ops` | laptop | `queue/pending/*.json` job specs + `config/runtime.json` |
| `ops-status` | desktop | `status.json` — heartbeat, running jobs, ledger, `at_a_glance` |
| `results` | desktop | `results/<job-id>/<policy>*` artifacts |

Exactly one writer per branch, so pushes are `--force-with-lease` and never merge.

## What changes from snek2's version

- **Every spec carries a required `project` field**, validated in `job.py`, and the daemon refuses
  anything that is not `snek3`. `ops` holds ~150 stale snek2 specs in `queue/pending/`; this is what
  stops one being dispatched by accident, and a future era inherits the guard.
- **One eval stage, not two.** The `training → closeout → HOF` chain becomes `training → stage B`, so
  the `auto_hof` path, the `-hof` id handling and the legacy phase branch all go.
- **`_ensure_viewer` goes** (~150 lines). The launcher owns the chart window.
- **`publish_results` gets a retry.** Without one, a failed push leaves the commit local while the
  ledger says `done` — indistinguishable from a pass that legitimately found nothing, and it once hid
  four 500-episode result files including a 98.2% checkpoint for hours.
- **A `failed` eval gets surfaced in `at_a_glance`.** Silently never retrying one cost snek2's batch
  46 wave 1 its whole measurement.

## What is carried unchanged, and why

Each of these is a documented incident rather than a preference:

- **The boot-id reattach.** A running record whose recorded boot id differs from
  `/proc/sys/kernel/random/boot_id` is marked **`interrupted`** — non-terminal, therefore relaunched,
  with a training resuming from its checkpoint. Before this existed the same situation read `done`,
  which published truncated arms as finished and silently consumed their measurements. Read
  `interrupted` as "lost wall clock, nothing else".
- **The wave barrier.** Trainings and evals never overlap, and nothing backfills until a wave drains.
- **`git_seconds` (600) apart from `poll_seconds` (30).** Work the box generates for itself starts in
  seconds while github traffic stays near 144 fetches a day. **A `status.json` up to 10 minutes old is
  a healthy daemon.**
- **`trigger`.** One ssh round trip that forces a fetch, dispatch and publish, and reports whether the
  daemon is polling at all — so one command both starts queued work and answers "did it start?".
- **`EVAL_RELEVANT_ENV`.** A wave is keyed on the settings that can reach a measurement of an
  already-trained checkpoint — shaping and reward knobs, not seeds or learning rates. Keying on the
  whole inherited env split one snek2 batch into three waves of 2/1/1 arms.
- **The memory bands.** The box has **15,030 MB** and memory is its binding constraint, not cores. It
  is a Ryzen 7 9700X: **8 physical cores, 16 SMT threads**, and its vec-eval optimum is **16 shards
  with one intra-op thread** — throughput climbs past the physical cores, and the cliff is at
  `cpu_count`, where 18 loses 6-10%.

## The rule that matters most when reading the box

**`git fetch` is not optional**, and leaving it out is the single most repeated mistake in this
project's history with the desktop. See [`../../CLAUDE.md`](../../CLAUDE.md) for the ladder.
