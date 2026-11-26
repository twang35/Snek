# Laptop Wi‑Fi: the stuck association, and the plan to stop it

**Status: diagnosed 2026-09-04; TWT ruled out the same evening (the router does not advertise it); laptop joined the 6 GHz radio 2026-09-04 18:05 (ch 69, ‑60 dBm, 864 Mb/s); first checks clean, verdict needs a day of load.** The laptop's Wi‑Fi periodically drops into a state
where a quarter to a half of packets vanish, at strong signal, until Wi‑Fi is switched off and on. It
happens every few hours, usually during heavy Claude Code plus training activity, and it is what a hung
`git fetch`, a stalled `rsync` to the desktop and a timed-out auto-mode classifier all look like from
inside a session. This page is the evidence, what was ruled out, and the steps to try, in order.

## 2026-11-25 episode: the loss is inside the router

**Status: located, not yet explained. The packets are lost in the router's handling of this one client;
what puts the router into that state is unknown.** Onset 06:54:37–06:54:58 PT (the recorder's 20 s
resolution), on AC power, load 2, 15 h into the association made by the 15:41 toggle the day before. It
stayed bad for over two hours without a toggle. Four captures plus a live run agree:

| test (bad state) | result | what it says |
|---|---|---|
| kernel echo replies counted vs replies `ping` got, 200 sent | 141 vs 140 | nothing is lost above the driver: GlobalProtect and the host stack are cleared |
| ping to the laptop's own address | 0% lost | same |
| ping to the desktop vs to the router | 18–44% vs 18–32% | every path through the router, not the router's own CPU |
| 1,400-byte vs 64-byte pings, 10/s | 50–88% vs 18–32% lost | loss grows steeply with packet size |
| 1,400-byte vs 32-byte pings, 1 every 1.1 s (live) | 72% vs 0% of 40 | the same, at its cleanest |
| driver during those large pings | `txFail=0`; received frames marked retry 1–8%; beacons 99% | every frame the laptop sent was acknowledged by the router, and the router was not retrying frames at the laptop |
| round trip | 40–100 ms typical, up to 670 ms | packets that survive are being held somewhere |
| ssh / scp to the desktop | hang for minutes | TCP is unusable at this loss on full-size segments |

A frame the laptop sent and the router acknowledged, whose reply never comes back and was never retried
over the air, was dropped or held inside the router. Loss that scales with packet size while the radio
statistics stay clean reads as a queue or limiter working in bytes on this client's traffic, not as radio
errors. The power-save reading of 2026-11-24 does not explain the size dependence and is demoted; the
host-path candidate is ruled out.

**The Wi‑Fi log shows nothing at the onset** (one routine auto-join retry 30 s earlier, which runs every
8 minutes). The cause is not visible from the laptop.

**The router** is a TP‑Link Archer BE600 (tri-band BE9700, Wi‑Fi 7), firmware 1.3.8 Build 20261119
rel.16680. The build date is newer than the fault (first diagnosed 2026-09-04), so that firmware did not
introduce it and has not fixed it.

**The router's system log (saved 09:41 PT) records nothing about Wi‑Fi clients at all**, so it neither
confirms nor contradicts the reading above: no line at the 06:54 onset or at the 11-24 episode. What it
does give:

| fact | from the log |
|---|---|
| last router boot | about 2026-11-23 03:02 PT (boot lines stamped with the default clock, then the first real timestamp). Both episodes came within 1.5–2.5 days of a fresh boot, so this is not a long-uptime fault |
| clients | 11, all on Wi‑Fi |
| HomeShield | no paid profile; its traffic-inspection (DPI) services were running until 09:37 |
| QoS | off since boot; **turned on by the user at 09:37:09 PT, 2026-11-25**, no rules set. Turning it on stopped the DPI services |

**Experiment 1, running: QoS on (no rules), from 09:37 PT 2026-11-25.** It changes how the router handles
every packet, so it is a fair first single change. Baseline is an episode roughly daily (11-24 ~14:00,
11-25 06:54) and "every few hours" in September. Read `~/wifi-diag/watch.log`: two clear days suggests it
helped, an episode with QoS on rules it out. Change nothing else on the router meanwhile.

**Router wireless settings (the user, 2026-11-25):** TWT off, OFDMA / MU‑MIMO off, Smart Connect off,
**Airtime Fairness on**. That removes the September suspects (OFDMA, MU‑MIMO, steering) and leaves one
feature that matches the measurements: Airtime Fairness is a per-client scheduler inside the router that
budgets each client's transmit time, so a client it has throttled loses packets in proportion to their
airtime (large ones first), has the survivors delayed in a queue, and shows clean radio statistics on its
own side. **Leading candidate, not proven.**

**Experiment 2, recommended in place of experiment 1: Airtime Fairness off, QoS back to off** (so exactly
one setting differs from the configuration that produced the episodes). Two clear days in
`~/wifi-diag/watch.log` supports it; an episode with it off rules it out. If the user keeps QoS on as
well, a clean result cannot be attributed to either.

**Experiment 2 running.** The recorder logged the link down 10:38:34–10:40:14 PT 11-25, read as the
router applying the Airtime Fairness change (the user confirmed it is off; QoS state not confirmed).
Check at 15:51 PT, 5 h 10 m in: no episode, 0 checks at ≥15% loss of ~940, 8 with any loss, average
round trip 4.7–6.2 ms per hour, 1,400 B pings 0/50 lost. **Not yet evidence:** the hours before the
06:54 onset looked the same, and episodes were 15–17 h apart. First meaningful read is the morning of
11-26; the verdict is two clear days (to ~10:40 PT 11-27).

**Check at 10:45 PT 11-26, 24 h 05 m in: held.** No episode, no drop, 0 checks at ≥15% loss of ~3,890,
19 with any loss, hourly average round trip 4.6–5.5 ms, 1,400 B pings 0/50 lost. This is past the
15–17 h gap between the earlier episodes, so it is the first real support for Airtime Fairness as the
cause; one more clear day completes the test. QoS state still not confirmed.

**15:15 PT 11-26: the laptop left the house**, still clean at 28.5 h (0 bad checks of 4,621). The
recorder now logs `away gw=<ip>` and never captures unless the gateway's hardware address is the home
router's (`HOME_MAC` in `wifi_watch.sh`; restarted, pid in `watch.pid`). The two-day clock counts only
hours at home, so the verdict moves later by the time away.

**Next, in order.**

1. **Still unknown on the router:** MLO on or off, and the laptop's row in the client table during an
   episode.
2. **Router settings that act per client, one at a time:** any QoS / device priority / bandwidth limit /
   parental or security profile on the laptop; airtime fairness; OFDMA and MU‑MIMO; Smart Connect and
   multi-link. Update the firmware first if one is offered.
3. **A router reboot instead of a laptop toggle, once:** if a reboot alone clears the laptop's bad state,
   that confirms the state lives in the router.
4. Ethernet at the desk remains the certain workaround; a watchdog that re-associates on detection is the
   automatic one.

The recorder's `counters` parsing in the first captures reads 0 (a parsing slip in the summary, the raw
files are right), and its desktop check now has a 45 s limit after hanging a capture for 25 minutes.

## 2026-11-24 episode: the Mac's radio layer does not see the loss

**Status: root cause not proven. The 6 GHz move did not stick (the laptop was back on channel 153), and the
fault recurred there. A recorder now captures the next episode automatically.**

| measured (all times PT) | bad state, ~14:17–15:40 | after the Wi‑Fi toggle, 15:41 |
|---|---|---|
| ping to the router, 10/s | 30.5% of 600 lost, in 100–600 ms runs with no period; RTT 4–113 ms, mean 35 | 0 of 300 lost; RTT 3–22 ms, mean 5 |
| channel, RSSI | 153 / 80 MHz, ‑50 to ‑57 dBm | the same |
| driver: transmit retries / failures | 20–30% / ~0 | 25–35% / 0 |
| driver: received frames marked retry | 2–4% | 2–3% |
| driver: beacons received | 99% | 99% |
| load, power | load 2–3, on battery, `powermode 1` (Low Power Mode, set for AC too) | the same |

The driver's 5‑second link records (`log show`, `LQM: rssi=`) are **the same in both states**, so the lost
packets are not failing as radio frames the Mac can count. They vanish either inside the router's
per-client state or in the Mac between the driver and the IP stack. The September reading ("802.11ax
feature interaction") is neither confirmed nor excluded by this.

| candidate | standing | evidence |
|---|---|---|
| router holds the laptop's frames as if it were asleep (power-save state out of step) | **leading, unproven** | RTT spreads up to one beacon interval (102 ms); loss without frame failures; cleared only by re-association; other clients fine |
| Mac host path (GlobalProtect network extension, Skywalk) dropping above the driver | open | not tested in the bad state; the kernel-counter test in the recorder settles it |
| channel contention on 153 | not the cause | channel busy 18–25% in both states; 294 neighbour beacons per 5 s is an aggravation only |
| AWDL, Bluetooth, background scans, CPU load | ruled out | AWDL 0 packets; no Bluetooth device connected, 0% airtime; one 7.7 s Duo Desktop scan only; load 2–3 |

**The recorder.** `~/wifi-diag/wifi_watch.sh` (read-only, outside the repo) pings the router every 20 s into
`~/wifi-diag/watch.log`; on two bad checks in a row it writes `~/wifi-diag/captures/<stamp>/` and posts a
notification that it is safe to toggle. Each capture answers the open question directly:

| file | what it decides |
|---|---|
| `counters_before/after.txt` vs `ping_gw.txt` | kernel echo replies counted vs replies `ping` received: equal to sent = lost above the driver (host); equal to received = lost at or below the radio |
| `ping_self.txt`, `ping_desktop.txt`, `desktop_view.txt` | host stack alone; another client through the same router; the other side's view |
| `ping_gw_slow.txt` vs `ping_gw.txt` | 1/s against 10/s: more loss when the radio is allowed to idle points at power save |
| `wifi_log.txt` and `watch.log` | the onset to within 20 s, and what the driver logged then |

**When it goes bad (the user's part).**

1. Do not toggle Wi‑Fi yet. Wait for the "Wi‑Fi fault recorded" notification, about two minutes after the
   link degrades (the capture itself takes ~90 s).
2. No notification after three minutes: `tail -3 ~/wifi-diag/watch.log`. No fresh lines means the recorder
   is not running (it does not survive a reboot or logout): start it with
   `nohup ~/wifi-diag/wifi_watch.sh >/dev/null 2>&1 &`, wait two minutes, then continue.
3. Toggle Wi‑Fi, and tell the agent a capture exists. It reads `~/wifi-diag/captures/<stamp>/` and the
   onset time in `watch.log`.

One capture decides between the two open candidates; a second episode confirms it before anything on the
router or the laptop is changed.

**Next, in order.** (1) Wait for one capture, toggling only after the notification. (2) If it says power
save: turn Low Power Mode off (`sudo pmset -a powermode 0`) and see whether episodes stop. (3) If it says
host: repeat with GlobalProtect disconnected. (4) Either way, read the router's own client table during an episode; Ethernet at
the desk is the certain workaround. **6 GHz is out** (the user, 2026-11-24: its range is too short to be
useful), so the fix has to work on 5 GHz. The older steps below still
stand as fallbacks.

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

**Superseded in part, 2026-11-24:** step 1 (6 GHz) and the recommendation built on it are withdrawn — the
user does not want the 6 GHz network because it does not reach far enough. Steps 2 onward stand as
fallbacks behind the 2026-11-24 plan above.

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
ssh -o ConnectTimeout=10 the-claw-den "ping -c 20 -i 0.2 $(ipconfig getifaddr en0)"   # the other side sees the same loss (the laptop's address changes; it was .236, then .102)
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
