#!/usr/bin/env bash
# Prepare the-claw-den for ssh from outside the home LAN (2026-09-04).
# Run from the laptop:   ssh the-claw-den 'bash -s' < snek3/desktop/harden_ssh.sh
# Idempotent. Needs passwordless sudo (the box has it). Does NOT touch the router:
# the port forward (WAN 2222 -> 192.168.0.79:22), the DHCP reservation for the
# desktop and the DDNS hostname are done at the BE600's admin page.
set -euo pipefail

CONF=/etc/ssh/sshd_config.d/10-snek-hardening.conf
LAN=192.168.0.0/24

echo "== sshd: key-only, no root, only claw — written, then asserted"
# Written in full rather than appended to, so the script does not depend on what an
# earlier setup left behind (review 2026-09-04: the first version only printed these).
sudo tee "$CONF" >/dev/null <<'SSHD'
# Snek: key-only SSH. Added 2026-08-13 when Tailscale was removed; rewritten by
# snek3/desktop/harden_ssh.sh, which asserts the effective values below before the
# box is exposed through the router. The console (the box has a monitor) is the
# recovery path if a key is lost.
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
PermitEmptyPasswords no
AllowUsers claw
MaxAuthTries 3
LoginGraceTime 20
X11Forwarding no
SSHD
sudo sshd -t
[ -s ~/.ssh/authorized_keys ] || { echo "FATAL: ~/.ssh/authorized_keys is empty — key-only sshd would lock you out"; exit 1; }
eff=$(sudo sshd -T)
fail=0
for want in 'passwordauthentication no' 'kbdinteractiveauthentication no' 'permitrootlogin no' \
            'pubkeyauthentication yes' 'permitemptypasswords no' 'allowusers claw' 'maxauthtries 3'; do
  if grep -qx "$want" <<<"$eff"; then echo "  ok   $want"; else echo "  FAIL $want"; fail=1; fi
done
# sshd_config.d is included first and the first value wins, so a 0*-named file could override us.
if ls /etc/ssh/sshd_config.d/ | grep -v '^10-snek-hardening.conf$' | grep -q .; then
  echo "  note: other files in sshd_config.d:"; ls /etc/ssh/sshd_config.d/ | grep -v '^10-snek-hardening.conf$' | sed 's/^/        /'
fi
[ "$fail" -eq 0 ] || { echo "FATAL: effective sshd config is not key-only; not touching the firewall"; exit 1; }

echo "== fail2ban: 5 failures in 10 min -> 1 h ban, LAN exempt"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fail2ban >/dev/null
printf '[sshd]\nenabled = true\nbackend = systemd\nmaxretry = 5\nfindtime = 10m\nbantime = 1h\nignoreip = 127.0.0.1/8 %s\n' "$LAN" \
  | sudo tee /etc/fail2ban/jail.d/sshd.local >/dev/null
sudo systemctl enable --now fail2ban >/dev/null
sudo systemctl restart fail2ban
sleep 2; sudo fail2ban-client status sshd | head -4

echo "== ufw: LAN trusted (keeps mDNS/avahi and the alias working), ssh rate-limited from anywhere"
sudo ufw --force reset >/dev/null
sudo ufw default deny incoming >/dev/null
sudo ufw default allow outgoing >/dev/null
sudo ufw allow from "$LAN" comment 'home LAN' >/dev/null
sudo ufw limit 22/tcp comment 'ssh, rate-limited' >/dev/null
sudo ufw --force enable >/dev/null
sudo ufw status verbose

echo "== reload sshd (existing sessions unaffected)"
sudo systemctl reload ssh
echo "done. Now: router port forward, DHCP reservation, DDNS; then the laptop ssh-config Match block."
