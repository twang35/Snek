# Setup runbook

One-time. After this the box runs unattended and is driven from the laptop over
git; SSH (over Tailscale) is only a backstop.

Assumptions: **Ubuntu Server**, a dedicated user **`snek`**, repo at
`/home/snek/Snek`, miniconda at `/home/snek/miniconda3`. Adjust paths in
`config/host.env` and `systemd/snek-runner.service` if you change these.

Legend: 🖥️ = run **on the desktop**, 💻 = run **on the laptop**, 🤖 = I can run
this over SSH once the box is reachable (or you can run it on the desktop).

---

## Part 1 — Desktop bootstrap (🖥️, physical, once)

Gets the box on the network with SSH + Tailscale so I can take over. **These are
the commands you run on the desktop.**

```bash
# 1. System packages
sudo apt update
sudo apt install -y git build-essential curl openssh-server

# 2. SSH server on
sudo systemctl enable --now ssh

# 3. Tailscale (secure reachability, no port-forwarding)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up          # follow the login URL; note the machine name it prints

# 4. Let the laptop's Claude SSH in: paste the LAPTOP's public key here.
#    (get it on the laptop with:  cat ~/.ssh/snek_desktop.pub  -- see Part 2)
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'PASTE_LAPTOP_PUBLIC_KEY_HERE' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 5. Passwordless sudo for this user (so I can install packages/services over SSH)
echo "$USER ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/snek-runner
sudo chmod 440 /etc/sudoers.d/snek-runner

# 6. Miniconda
curl -fsSLo /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/mc.sh -b -p $HOME/miniconda3

# 7. Deploy key: lets the desktop PUSH to GitHub (status + results branches).
ssh-keygen -t ed25519 -N '' -f ~/.ssh/snek_deploy -C snek-desktop-deploy
cat ~/.ssh/snek_deploy.pub
#    -> Add that key at github.com/twang35/Snek > Settings > Deploy keys,
#       "Add deploy key", ALLOW WRITE ACCESS. Then wire git to use it:
cat >> ~/.ssh/config <<'EOF'
Host github-snek
  HostName github.com
  User git
  IdentityFile ~/.ssh/snek_deploy
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

Then tell me the **Tailscale machine name** from step 3. That's all I need — I do
the rest over SSH. (Or keep going below and run it yourself.)

---

## Part 2 — Create the bus branches (💻, laptop, once)

I run these from the laptop checkout. `ops` carries the whole tree (it holds the
queue + runtime.json); `ops-status` and `results` are orphan branches so they
don't drag the codebase along.

```bash
# laptop public key for Part 1 step 4 -- a dedicated key, already generated as
# ~/.ssh/snek_desktop (kept separate from the YubiKey so SSH needs no touch).
# The laptop connects with `ssh -i ~/.ssh/snek_desktop` (a Host alias gets added
# to ~/.ssh/config once the Tailscale name is known).
cat ~/.ssh/snek_desktop.pub

git branch ops master && git push origin ops
git switch --orphan ops-status && git commit --allow-empty -m "init ops-status" && git push origin ops-status
git switch --orphan results   && git commit --allow-empty -m "init results"   && git push origin results
git switch master
```

---

## Part 3 — Install the runner (🤖 over SSH, or 🖥️ on the desktop)

```bash
# git identity -- REQUIRED, or the daemon's commits to the bus branches fail
# silently and nothing is ever published.
git config --global user.name  "snek-runner"
git config --global user.email "snek-runner@$(hostname)"

# clone via the deploy remote so pushes use the deploy key
git clone github-snek:twang35/Snek.git /home/claw/Snek
cd /home/claw/Snek

# system libs for opencv's GUI (the live chart window)
sudo apt-get install -y libgl1 libglib2.0-0

# conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f snek2/desktop/environment.yml
conda activate snek
python -c "import tensorflow, tf_agents, cpprb, pygame, imageio, pyformulas, cv2; print('env OK')"

# worktrees for the two desktop-written branches (outside the main checkout)
git fetch origin ops-status results
git worktree add -B ops-status /home/claw/snek-bus/status  origin/ops-status
git worktree add -B results    /home/claw/snek-bus/results origin/results

# host config -- example already has this box's paths + DISPLAY/XAUTHORITY.
# Adjust XAUTHORITY if the graphical session's cookie path differs (see the
# comment in host.env.example for how to find it).
cp snek2/desktop/config/host.env.example snek2/desktop/config/host.env

# systemd service (daemon runs on base python; jobs use the env python)
sudo cp snek2/desktop/systemd/snek-runner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now snek-runner
systemctl status snek-runner --no-pager
```

Live chart windows appear on the desktop's monitor via the `DISPLAY`/`XAUTHORITY`
in `host.env`. Jobs still run and save their PNG charts if the display is
unavailable (the window is best-effort).

---

## Part 4 — Verify (💻 laptop)

Queue a smoke job and watch it run, without touching the desktop:

```bash
git checkout ops
cp snek2/desktop/queue/examples/smoke.json snek2/desktop/queue/pending/smoke-1.json
git add snek2/desktop/queue/pending/smoke-1.json && git commit -m "smoke" && git push origin ops
git checkout master

# within ~1 poll, status shows it running, then done:
git fetch origin ops-status && git show origin/ops-status:status.json
```

Expected: `smoke-1` appears under `running`, then in `ledger` as `done`; the
daemon self-terminated it at `max_steps`. Then remove the smoke file from
`queue/pending/` on `ops` so it isn't reconsidered.

---

## Keeping the runner code current

The daemon runs the code on the desktop's `master` checkout. To ship a runner
change: merge it to `master`, then on the desktop `git -C /home/snek/Snek pull`
and `sudo systemctl restart snek-runner`. The restart won't disturb running
trainings (detached; `KillMode=process`).

## If the box wedges (the one case git-only can't fix)

SSH in over Tailscale: `ssh snek@<tailscale-name>`, then
`journalctl -u snek-runner -n 100 --no-pager` and
`sudo systemctl restart snek-runner`.
