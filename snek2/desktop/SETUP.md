# Setup runbook

> ## ✅ Already done — this box is live
>
> **`the-claw-den` has been set up and verified (2026-08-08).** The daemon is
> `active`, the queue drains, and a smoke job has completed end to end. **You do not
> need to run anything below** unless you are rebuilding the box or adding a second
> one. For day-to-day use go to [`README.md`](README.md).
>
> | | |
> |---|---|
> | host | `the-claw-den`, reached as `the-claw-den.local` (mDNS) on the home LAN |
> | user | **`claw`** |
> | reach it | `ssh the-claw-den` — via the [`~/.ssh/config` alias](#laptop-side-ssh-access-and-how-to-rebuild-it) |
> | hardware | Ryzen 7 9700X, 8c/**16t**, **15,030 MB RAM** (`free -m`), Ubuntu 24.04 |
> | repo | `/home/claw/Snek` |
> | conda | `/home/claw/miniconda3`, env `snek` |
> | daemon | `systemctl status snek-runner` (runs on **base** python, jobs use the env python) |
>
> **Memory, not cores, is the binding constraint** — see
> [measured capacity](README.md#measured-capacity--memory-is-the-limit).

One-time. After this the box runs unattended and is driven from the laptop over
git; SSH is only a backstop, and since 2026-08-13 it reaches the box **over the
home LAN only** — see [Laptop-side SSH access](#laptop-side-ssh-access-and-how-to-rebuild-it).

Assumptions below: **Ubuntu**, a dedicated user, its home holding the repo and
miniconda. **On `the-claw-den` that user is `claw`**, so the paths are
`/home/claw/Snek` and `/home/claw/miniconda3`; the commands use `$HOME` and
`$USER` where they can so they work for any user name. Adjust `config/host.env`
and `systemd/snek-runner.service` to match whatever you pick.

Legend: 🖥️ = run **on the desktop**, 💻 = run **on the laptop**, 🤖 = I can run
this over SSH once the box is reachable (or you can run it on the desktop).

---

## Part 1 — Desktop bootstrap (🖥️, physical, once)

Gets the box on the network with SSH so I can take over. **These are the commands
you run on the desktop.**

```bash
# 1. System packages
sudo apt update
sudo apt install -y git build-essential curl openssh-server avahi-daemon

# 2. SSH server on
sudo systemctl enable --now ssh

# 3. mDNS, so the laptop can find the box as <hostname>.local without a static
#    IP or any port-forwarding. Ubuntu desktop has avahi already; just confirm.
sudo systemctl enable --now avahi-daemon
systemctl is-active avahi-daemon          # must print: active
hostname                                  # the name the laptop will use, + .local

# 4. Let the laptop's Claude SSH in: paste the LAPTOP's public key here.
#    (get it on the laptop with:  cat ~/.ssh/snek_desktop.pub  -- see Part 2)
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'PASTE_LAPTOP_PUBLIC_KEY_HERE' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 4b. Key-only SSH. MUST come after step 4 -- disabling passwords before the
#     laptop's key is in authorized_keys locks out every remote login.
#     `sshd -t` validates before the reload, so a typo cannot lock you out either.
sudo tee /etc/ssh/sshd_config.d/10-snek-hardening.conf >/dev/null <<'EOF'
PasswordAuthentication no
PermitRootLogin no
EOF
sudo sshd -t && sudo systemctl reload ssh
sudo sshd -T | grep -E '^(passwordauthentication|permitrootlogin)'   # expect: no, no

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

Then tell me the **hostname** from step 3 — `<hostname>.local` is all I need, and I
do the rest over SSH. (Or keep going below and run it yourself.)

---

## Part 2 — Create the bus branches (💻, laptop, once)

I run these from the laptop checkout. `ops` carries the whole tree (it holds the
queue + runtime.json); `ops-status` and `results` are orphan branches so they
don't drag the codebase along.

```bash
# laptop public key for Part 1 step 4 -- a dedicated key, already generated as
# ~/.ssh/snek_desktop (kept separate from the YubiKey so SSH needs no touch).
# Set up the Host alias too: see "Laptop-side SSH access" below.
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
git clone github-snek:twang35/Snek.git $HOME/Snek
cd $HOME/Snek

# system libs for opencv's GUI (the live chart window)
sudo apt-get install -y libgl1 libglib2.0-0

# conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f snek2/desktop/environment.yml
conda activate snek
python -c "import tensorflow, tf_agents, cpprb, pygame, imageio, pyformulas, cv2; print('env OK')"

# worktrees for the two desktop-written branches (outside the main checkout)
git fetch origin ops-status results
git worktree add -B ops-status $HOME/snek-bus/status  origin/ops-status
git worktree add -B results    $HOME/snek-bus/results origin/results

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

## Laptop-side SSH access (and how to rebuild it)

**Tailscale was removed on 2026-08-13** (not an approved app on the work laptop). It
was only ever the transport — name resolution and NAT traversal — so the swap to
plain OpenSSH over the LAN changed no command. What it did cost is **off-LAN shell
access**: see [Reaching the box](README.md#reaching-the-box).

Two things live outside this repo and are **not backed up**: `~/.ssh/config` (whose
other entries are all machine-generated `DO NOT EDIT` blocks that Airbnb tooling can
regenerate) and the private key `~/.ssh/snek_desktop`. So this section is the
canonical copy — everything needed to get back in is here.

### The alias

Restores in one command:

```bash
cat >> ~/.ssh/config <<'EOF'
Host the-claw-den
  HostName the-claw-den.local
  HostKeyAlias the-claw-den
  User claw
  IdentityFile ~/.ssh/snek_desktop
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

Three lines are load-bearing, not style:

- **`HostKeyAlias the-claw-den`** pins the `known_hosts` lookup to the alias rather
  than the real hostname, so the existing entry keeps working and a later change of
  address never re-triggers host-key verification.
- **`IdentitiesOnly yes`** stops ssh offering the YubiKey first — `snek_desktop`
  exists so SSH needs no touch.
- **`HostName the-claw-den.local`** is mDNS, so it survives the desktop's DHCP lease
  changing. The box is on **Wi-Fi** with a dynamic lease, so its address is not stable.

With the stanza in place **both call shapes work** — `ssh the-claw-den` and the older
explicit `ssh -i ~/.ssh/snek_desktop claw@the-claw-den` that appears throughout these
docs. That is why the transport change rewrote no commands.

### If `~/.ssh/config` is gone — no-config fallback

Fully self-contained, every option the alias supplies given explicitly:

```bash
# mDNS name, needs no config file at all
ssh -i ~/.ssh/snek_desktop -o HostKeyAlias=the-claw-den -o IdentitiesOnly=yes claw@the-claw-den.local

# if mDNS is also unavailable, the literal address
# (Wi-Fi DHCP -- was 192.168.0.79 on 2026-08-13, so confirm before trusting it)
ssh -i ~/.ssh/snek_desktop -o HostKeyAlias=the-claw-den -o IdentitiesOnly=yes claw@192.168.0.79
```

`-o HostKeyAlias=the-claw-den` is what lets one `known_hosts` entry serve every form.
Dropping it costs only a first-connection prompt, not a failure. Add `-F /dev/null` to
prove a command truly depends on no config file.

If neither name nor address works, find the box from the laptop with
`dscacheutil -q host -a name the-claw-den.local`, or `arp -a | grep 192.168.0`.

### If the key is gone — the irreplaceable half

**`~/.ssh/snek_desktop` is the one thing no fallback above can work around**, because
the desktop authorises that specific key. If `~/.ssh` is wiped, the key goes with it
and every command here fails identically, which reads like a network fault but is not.
**There is no password fallback** — the box is key-only since 2026-08-13
(`sshd_config.d/10-snek-hardening.conf`, Part 1 step 4b).

Recovery is a fresh keypair installed from the desktop's **physical console** — the box
has a monitor and keyboard, and that is the true out-of-band path:

```bash
ssh-keygen -t ed25519 -N '' -f ~/.ssh/snek_desktop -C snek-laptop   # on the laptop
cat ~/.ssh/snek_desktop.pub                                        # then type/paste it
# at the desktop console, as claw:
#   echo '<that public key>' >> ~/.ssh/authorized_keys
```

Two rules that keep the console from being the *only* way back:

- **Never commit the private key** to this repo. The public key is harmless; the
  private key is not, and this repo is on GitHub.
- **Keep a copy of `~/.ssh/snek_desktop` somewhere you control** — a password manager
  or an encrypted backup. This is a manual step nothing in the repo can do for you,
  and it is the difference between a one-minute fix and a trip to the desk.

## Keeping the runner code current

The daemon runs the code on the desktop's `master` checkout. To ship a runner
change: merge it to `master`, then on the desktop **hard-reset to origin** and
restart. The checkout is a pull-only mirror, so a plain `git pull` can fail with
"divergent branches" — reset instead of pull:

```bash
ssh -i ~/.ssh/snek_desktop claw@the-claw-den \
  'git -C ~/Snek fetch origin master && git -C ~/Snek reset --hard origin/master && sudo systemctl restart snek-runner'
```

The restart won't disturb running trainings (detached; `KillMode=process`).

## If the box wedges (the one case git-only can't fix)

```bash
ssh -i ~/.ssh/snek_desktop claw@the-claw-den \
  'journalctl -u snek-runner -n 100 --no-pager; sudo systemctl restart snek-runner'
```

## What went wrong during the real setup, so it isn't re-debugged

Both are **fixed** — recorded because each cost a failed job and neither is
obvious from a traceback.

| symptom | cause | fix |
|---|---|---|
| daemon `active` but `status.json` never published | `claw` had no git `user.name`/`email`, so the daemon's bus commits failed — and the bus helper ran git with `check=False`, swallowing the error | set git identity (Part 3); `gitbus` now uses `check=True` so a commit failure surfaces (`057bd17`) |
| `smoke-1`/`smoke-2`: `ModuleNotFoundError` for `imageio` / `cv2` | env spec lacked `imageio`, `opencv-python`, `pyformulas` (pulled in transitively by `under_the_hood`/`pyformulas`) | `environment.yml` pins all three, plus system `libgl1 libglib2.0-0` for opencv's GUI |
| bench trainers died `Aborted (core dumped)` when launched with no `DISPLAY` | opencv's GUI build calls `abort()` **natively** when it can't reach a display — a SIGABRT the Python best-effort guard cannot catch | jobs get `DISPLAY=:0` + `XAUTHORITY` from `host.env` (the box has a monitor); the best-effort window only saves you if the display is genuinely gone |

**The verification line in Part 3 is what catches this class**, so don't skip it —
it imports every module a job needs, including the transitive ones:

```bash
python -c "import tensorflow, tf_agents, cpprb, pygame, imageio, pyformulas, cv2; print('env OK')"
```
