# Laptop Wi‑Fi: the stuck association, and the plan to stop it

**Status: diagnosed 2026-09-04; TWT ruled out the same evening (the router does not advertise it); next step is the 6 GHz radio.** The laptop's Wi‑Fi periodically drops into a state
where a quarter to a half of packets vanish, at strong signal, until Wi‑Fi is switched off and on. It
happens every few hours, usually during heavy Claude Code plus training activity, and it is what a hung
`git fetch`, a stalled `rsync` to the desktop and a timed-out auto-mode classifier all look like from
inside a session. This page is the evidence, what was ruled out, and the steps to try, in order.

## Symptom

| what | value in the bad state | value after a Wi‑Fi toggle, same load |
|---|---|---|
| ping to the router (192.168.0.1) | 25–55% loss, RTT 1 ms → 30–130 ms in 1–2 s bursts | 0% loss, 3–5 ms |
| ping from the desktop to the laptop | 39% loss | 0% loss |
| ping from the desktop to the router | 0% loss, 5 ms | 0% loss |
| RSSI / noise | ‑53 to ‑56 dBm / ‑90 dBm | unchanged |
| Tx rate / MCS | 612 → 154 → 432 Mb/s, MCS 5–6 | 544+ Mb/s |
| `wdutil` faults, recoveries, link tests (last hour) | none | — |

The router and the LAN are fine (the desktop, on Wi‑Fi to the same router, was clean the whole time).
Loopback pings on the laptop are perfect, so the kernel's IP stack is fine. The impairment is the
laptop's own association with the access point, and the driver does not know it is broken.

## Ruled out

| candidate | why not |
|---|---|
| CPU starvation | loss persisted after load fell from 38 to 12 with 15% idle; and was 0% right after the toggle at load 23 with 8 trainers running |
| memory / swap thrash | same: 3.5 GB in swap, 14 MB free at the worst, and the link stayed bad after that eased. Plausible *trigger*, not the cause |
| socket exhaustion | 122 established TCP connections, mbufs at 20% of the pool, driver queue dropped 12 packets in 21 days |
| DNS / mDNSResponder | resolution worked (slowly, because packets were being lost); daemons all at 0% CPU |
| AWDL / AirDrop / Continuity | AWDL disabled, 223 packets ever; Bluetooth off |
| battery power saving | on AC |
| VPN / proxy | five `utun` interfaces but the default route is `en0` direct; no proxy |
| interface errors | `Ierrs`/`Oerrs` both 0 |
| Airbnb git-wrapper telemetry curls | six were stuck in `SYN_SENT` at once; they pile up *because* the link is bad, and each gives up after 10 s |

## Best reading

A per-client link state the Mac and the router negotiate badly, most often one of the 802.11ax features
(Target Wake Time, OFDMA, MU-MIMO), or client steering (802.11k/v). It survives until reassociation, which
is why the toggle fixes it and nothing else does. The desktop is spared by using a different vendor's
driver (Qualcomm WCN785x, Wi‑Fi 7), not by being an older client. Two aggravating factors: channel 153 at
80 MHz overlaps **nine** neighbouring 80 MHz networks (channel busy 32%), and load spikes to 38 on 14 cores
with the machine in swap are the likely trigger for the link to fall into the bad state in the first place.

## Steps, in order

Each step is one change, then wait for the next occurrence (a few hours of normal use) before the next.

| # | where | change | how to tell it worked |
|---|---|---|---|
| ~~1~~ | router | ~~disable Target Wake Time (TWT)~~ **Already off** — the 5 GHz beacon's HE MAC capabilities advertise no TWT responder, requester, broadcast or flexible TWT (read 2026-09-04, method below). Not the cause | — |
| 1 | laptop | **join the router's 6 GHz radio** (channel 69, 160 MHz, WPA3, the only 6 GHz network in range). The laptop's card (Broadcom 0x4388, Wi‑Fi 6E) supports it. Zero co-channel neighbours instead of nine, a different radio and channel, and if it is its own SSID, no band steering for the laptop. Same LAN, same subnet — `rsync`/`ssh`/mDNS to the desktop unchanged | `system_profiler SPAirPortDataType` shows `Channel: 69 (6GHz…)`; RSSI still better than ‑65 dBm at the desk; no stuck state in a day |
| 2 | router | if staying on 5 GHz: move off channel 153 to **36–48** (one neighbour there instead of nine); fixed channel, not auto; 40 MHz if 80 still struggles | new channel in `system_profiler`; router-side channel-busy drops |
| 3 | router | disable **OFDMA**, then **band steering / 802.11k/v/r** (the beacon carries an RM Enabled Capabilities element, so 802.11k is on) if 1–2 did not end it | same as 1 |
| 4 | laptop | **watchdog**: a launchd job that pings the gateway every 30 s and cycles Wi‑Fi (`networksetup -setairportpower en0 off/on`) after ≥60 s of ≥40% loss while the interface reports `status: active`. Automates the manual fix; does not remove the cause | the log shows it firing and the outage lasting seconds, not until someone notices |
| 5 | laptop | **wired**: a USB-C Ethernet adapter. Removes the problem outright and makes `rsync`/`ssh` to the desktop reliable | — |
| 6 | laptop | **reduce the trigger**: run `laptop_batch` under `nice -n 10`, and keep trainers + eval workers ≤ cores − 2 (14 cores: 8 trainers + 8 workers is 16 pegged processes). Also `SNEK_EVAL_WORKERS` could drop to 6 | load average stays under ~14 during a batch; no swap growth |

Recommendation: do 1 (6 GHz) first — it is a laptop-side change, reversible in a click, and it removes the
channel contention outright. If the stuck state follows the laptop onto 6 GHz, the cause is the Mac ↔
router 802.11ax negotiation rather than the channel, and 3 is next. Do 4 regardless, since it is the only step
that helps even if the cause turns out to be something else. 5 is the certain fix if nothing else is.

**6 GHz caveats.** Range and wall penetration are worse than 5 GHz; at ‑55 dBm on 5 GHz the desk is close
enough, but check the 6 GHz RSSI after joining. If the router runs Smart Connect (one SSID across bands),
the router chooses the band and the Mac cannot pin it — give the 6 GHz radio its own SSID so the laptop is
pinned there; the desktop stays on 5 GHz. The bands are one bridged LAN inside the router, so the laptop
keeps its 192.168.0.x address, the desktop reaches it the same way, and `the-claw-den` mDNS still resolves.

## Reading what the router actually advertises

The router's admin page says what is *configured*; the beacon says what clients are *offered*. The desktop
has passwordless `sudo` and `wpa_cli`, so from the laptop:

```
ssh the-claw-den 'sudo -n wpa_cli -i wlp11s0 scan >/dev/null; sleep 4; sudo -n wpa_cli -i wlp11s0 bss 72:7f:f0:55:99:f3' \
  | grep -E '^(freq|ie)='
```

Then decode the `ie=` hex: walk the elements (id, length, body); element 255 with extension id 35 is HE
Capabilities, and its first body byte is HE MAC capabilities byte 0 — bit 1 TWT requester, bit 2 TWT
responder; byte 2 bit 4 broadcast TWT. On 2026-09-04 that byte was `0x01`: **no TWT of any kind**, which
matches the router UI. The same dump showed 802.11k (element 70) present, an EHT Operation element (Wi‑Fi 7),
and BSS Load at 9 stations and 29–35% channel utilisation. The desktop's `wpa_cli` only listed the 5 GHz BSS,
so the 6 GHz radio's beacon has to be read from the laptop (`sudo wdutil info` while joined to it).

## Diagnosing it next time, in one minute

From inside a session, without sudo:

```
ping -c 20 -i 0.2 192.168.0.1                                       # loss with strong RSSI = this
system_profiler SPAirPortDataType | grep -E 'Signal|Transmit Rate|MCS|Channel'
ssh -o ConnectTimeout=10 the-claw-den 'ping -c 20 -i 0.2 192.168.0.236'   # the other side sees the same loss
```

From a real terminal (the `!` prefix cannot prompt for a password), before toggling, for the driver's view:

```
sudo wdutil info
```

Then toggle Wi‑Fi and re-run the first ping under the same load. Clean = stuck association, as above.

**Three traps this produces inside a session.** `git fetch` fails with `RPC failed; curl 56` (retry works,
not "GitHub is down"); an `rsync`/`ssh` to the desktop hangs rather than failing (the tool now times out
and carries on; not "off-LAN"); and the auto-mode safety classifier times out, so Bash *writes* stall
while read-only commands still run — use the Edit tool for edits until the link is back.

## Related

- Memory note: `laptop-wifi-stuck-association` (the agent's own reminder of the above)
- [`../../CLAUDE.md`](../../CLAUDE.md), "There are two compute hosts": the ladder for deciding the desktop is unreachable — a stuck laptop link is now the first thing to rule out
