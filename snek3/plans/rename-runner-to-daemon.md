# Rename `runner` to `daemon`

**Status: planned 2026-09-05, not started. Runs after the working tree is clear** of the other
session's edits (`desktop/runner/runner.py`, `trigger.py`, `tools/scheduler.py`, `live_runs.py`,
`laptop_status.py`, `eta.py` and their tests). A rename over a dirty tree is a merge conflict with
ourselves.

## Why

Every doc already calls it "the daemon" (`CLAUDE.md`, `desktop/README.md`, `plans/scheduler.md`,
the skills), and since the scheduler took the box over (2026-09-05) the process *runs* nothing: it
polls the git bus, writes the hold marker, starts one scheduler, publishes status. `runner` was the
name from when it launched the arms itself. One name in the code and another in every sentence about
it is a tax on every reader, so the code takes the name the prose already uses.

**No compatibility shim, no alias, no old name kept anywhere** (user, 2026-09-05: "I'd much rather
have the codebase clean than incur lots of cruft keeping old things around that nobody uses"). The
service, the state directory and the env vars are renamed with the package, in one cut.

## The rename, in full

| today | after |
|---|---|
| `desktop/runner/` (package) | `desktop/daemon/` |
| `desktop/runner/runner.py` (module, `python -m runner.runner`) | `desktop/daemon/daemon.py` (`python -m daemon.daemon`) |
| `from desktop.runner import gitbus` / `import runner as daemon` (`tools/laptop_status.py`, `tools/scheduler.py`, tests) | `from desktop.daemon import gitbus` / `from desktop.daemon import daemon` |
| `from runner.job import ...` (`tools/sweep_specs.py`) | `from daemon.job import ...` |
| `-m runner.deploy`, `-m runner.trigger` (`desktop/deploy`, `desktop/trigger`, `_deploy` in the daemon) | `-m daemon.deploy`, `-m daemon.trigger` |
| `_runner_changed` and its diff paths `snek3/desktop/runner` | `_daemon_changed`, `snek3/desktop/daemon` |
| `desktop/systemd/snek3-runner.service`, `trigger.UNIT_NAME = 'snek3-runner'` | `snek3-daemon.service`, `'snek3-daemon'` |
| `SNEK_RUNNER_HOST_ENV` (unit file, `config.py`, `trigger.py`, the daemon's env allow-list) | `SNEK_DAEMON_HOST_ENV` |
| `SNEK_RUNNER_PYTHON` (`desktop/deploy`, `desktop/trigger`) | `SNEK_DAEMON_PYTHON` |
| `~/.snek3-runner/` (`LEDGER_PATH`, `LOG_DIR` in `host.env`, not in git; `host.env.example`) | `~/.snek3-daemon/` |
| `tests/test_desktop_runner.py` | `tests/test_daemon.py` |
| ~280 mentions in `desktop/README.md`, `plans/scheduler.md`, `plans/pytorch-port.md`, `docs/findings.md`, `skills/desktop-deploy`, `skills/desktop-batch`, `tools/closeout.py`, `evaluate.py` docstrings, root `CLAUDE.md` ("snek3-runner"), memory `desktop-code-deploy.md` | `daemon` |

Not renamed: `wall_runner` in `test_observation_layout.py` and "test-runner" in `test_reward_preset.py`
are different words. History (`plans/scheduler.md` §0 incidents, `docs/findings.md` dated entries)
may keep the old spelling *inside a quoted command that was typed on that date*; everything stating
what is true now changes.

## The one hazard: the box restarts into the new name

The daemon deploys itself: `queue_action deploy` makes it fast-forward and, because
`desktop/runner/` changed, exit for systemd to restart it. After this commit systemd would restart
`python -m runner.runner` from a tree that has no `runner/` — **a crash loop every 10 s, and only sudo
gets out of it.** The unit file is root-owned, so the box's half is a script the user runs
(`~/.ssh` memory: root changes over ssh go in a script the user types). So the order is:

1. **Laptop:** rename, update, tests green, commit, push master. Do **not** queue a deploy yet.
2. **Box, user-typed** (`! ssh the-claw-den 'bash -s' < snek3/desktop/systemd/rename-to-daemon.sh`,
   the script is committed with the rename and does all of this):
   - `sudo systemctl stop snek3-runner` — `KillMode=process`, so the scheduler and its arms keep
     running; the daemon re-adopts the scheduler by pid when it returns.
   - `mv ~/.snek3-runner ~/.snek3-daemon`; `sed -i 's/\.snek3-runner/.snek3-daemon/' desktop/config/host.env`.
   - `cd ~/Snek && git fetch && git merge --ff-only origin/master` (the tree has `snek3/desktop/daemon/`
     now; `desktop/deploy` itself is on the new name, so run git directly rather than through it).
   - `sudo cp snek3/desktop/systemd/snek3-daemon.service /etc/systemd/system/ && sudo systemctl
     disable --now snek3-runner; sudo rm /etc/systemd/system/snek3-runner.service; sudo systemctl
     daemon-reload && sudo systemctl enable --now snek3-daemon`.
   - `systemctl is-active snek3-daemon && Snek/snek3/desktop/trigger` — exit 0 and both boxes' glance.
3. **Laptop:** `git fetch origin ops-status` and check `head` is the rename sha; the running scheduler
   (old code, untouched) and its arms are still in `running`.

**Pick the moment:** the box's arms and passes do not care, but the daemon is down for the minute the
script takes, so nothing on `ops` is dispatched then. Any time is fine; between waves is tidiest.
The laptop needs no box step — `tools.laptop_status` imports the new package name and the next
scheduler start picks it up; the running scheduler is old code until it is next started (as for any
scheduler change, `skills/desktop-deploy`).

## Order of work on the laptop

1. `git mv desktop/runner desktop/daemon && git mv desktop/daemon/runner.py desktop/daemon/daemon.py`;
   `git mv desktop/systemd/snek3-runner.service desktop/systemd/snek3-daemon.service`;
   `git mv tests/test_desktop_runner.py tests/test_daemon.py`.
2. `grep -rn 'runner' --include='*.py' --include='*.md' --include='*.service' --include='*.example'
   . ../CLAUDE.md desktop/deploy desktop/trigger desktop/queue_action` and fix every hit that is this
   daemon, by hand, not a blind `sed` (the two false friends above, and prose that reads "the daemon
   runs" needs no "runner" at all). The `_StubJob in runner.py` comment in `job.py` and the launch.py
   comment name the module: `daemon.py`.
3. `SNEK_RUNNER_*` → `SNEK_DAEMON_*` in code, unit, wrappers, `host.env.example`, README.
4. Write `desktop/systemd/rename-to-daemon.sh` (the box script above; idempotent, `set -euo pipefail`,
   prints each step). Delete it in a later docs commit once it has run — it is a migration, not a tool.
5. Tests: rename the file, rewrite imports and any assertion on a path string (`'snek3/desktop/runner'`
   in the restart-iff test, `UNIT_NAME`, the env allow-list) to the new names. Run the whole suite.
   `tests/test_module_layering.py` and `test_desktop_deploy.py` import the package by name.
6. `python -m tools.laptop_status --help`-level smoke: `PYTHONPATH=. python -c 'from desktop.daemon import
   daemon, gitbus'`, and `desktop/trigger` from the laptop (it only needs `daemon.trigger` importable
   locally before it ssh-es).
7. Show the diff, wait for approval, commit as one change, push. Then the box script.

## Verification

- `grep -rn 'runner' snek3 --include='*.py' --include='*.md' --include='*.sh' --include='*.service'
  | grep -v 'wall_runner\|test-runner'` returns only dated history.
- Suite green on the laptop; `ssh the-claw-den 'Snek/snek3/desktop/trigger'` exit 0 with the new unit;
  `journalctl -u snek3-daemon -n 20` shows one start, no restart loop; `ls ~/.snek3-daemon/logs` has
  the daemon's new log; `ops-status` `head` = rename sha; a `queue_action restart` round-trips.
- `ls /etc/systemd/system | grep snek3` shows only `snek3-daemon.service`; `~/.snek3-runner` is gone.
