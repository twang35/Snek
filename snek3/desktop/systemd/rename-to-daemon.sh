#!/usr/bin/env bash
# One-time box-side half of `plans/rename-runner-to-daemon.md`: snek3-runner -> snek3-daemon.
# Run as claw on the-claw-den, from the laptop:  ssh the-claw-den 'bash -s' < snek3/desktop/systemd/rename-to-daemon.sh
# Needs sudo for the unit swap. Idempotent: every step checks before it acts. KillMode=process, so the
# scheduler and its arms keep running through the stop; the new daemon re-adopts the scheduler by pid.
set -euo pipefail
cd ~/Snek
say() { printf '\n== %s\n' "$*"; }

say "stop the old unit (if it is installed)"
if systemctl list-unit-files snek3-runner.service --no-legend | grep -q snek3-runner; then
    sudo systemctl disable --now snek3-runner
fi

say "state dir ~/.snek3-runner -> ~/.snek3-daemon"
if [ -d ~/.snek3-runner ] && [ ! -e ~/.snek3-daemon ]; then mv ~/.snek3-runner ~/.snek3-daemon; fi
ls -d ~/.snek3-daemon

say "host.env paths"
sed -i 's/\.snek3-runner/.snek3-daemon/g' snek3/desktop/config/host.env
grep -n 'snek3-daemon' snek3/desktop/config/host.env

say "fast-forward master (git directly: desktop/deploy is itself on the new name)"
git fetch -q origin && git merge --ff-only origin/master
test -d snek3/desktop/daemon

say "install snek3-daemon.service, drop snek3-runner.service"
sudo cp snek3/desktop/systemd/snek3-daemon.service /etc/systemd/system/
sudo rm -f /etc/systemd/system/snek3-runner.service
sudo systemctl daemon-reload
sudo systemctl enable --now snek3-daemon
sleep 3
systemctl is-active snek3-daemon
ls /etc/systemd/system | grep snek3

say "trigger: the daemon polls and publishes now"
snek3/desktop/trigger
