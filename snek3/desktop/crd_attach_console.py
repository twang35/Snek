#!/usr/bin/env python3
"""Make Chrome Remote Desktop share the monitor's own screen instead of a hidden virtual desktop.

Run on the box as root, and again after every chrome-remote-desktop package upgrade (the upgrade
rewrites the script and silently brings the virtual desktop back):

    ssh the-claw-den 'sudo python3 -' < snek3/desktop/crd_attach_console.py
    ssh the-claw-den 'sudo systemctl restart chrome-remote-desktop@claw'

What it does, and why: on Linux the CRD host script starts its own Xorg on a spare display and a
desktop session inside it, so a remote login lands on a screen nobody can see on the monitor. The
user's 2023 copy of the script (Downloads, "works for 22.04 in Jan 2023") commented out those two
launches and pointed DISPLAY at the console's display, and the host then captured the real screen.
This does the same to the current script (checked against 155.0.8059.19), plus one more line the
2023 version did not need: the main loop now launches the host only after it has started a server,
so that condition is relaxed to "no host yet". XAUTHORITY is gdm's cookie for :0, which it writes
under the user's runtime dir. Idempotent: a second run finds the marker and changes nothing.
Undo: copy the .orig back over the script and restart the service.
"""
import os
import shutil
import sys

PATH = "/opt/google/chrome-remote-desktop/chrome-remote-desktop"
MARKER = "attach-console 2026-10-03"

OLD_LAUNCH = """    self._launch_server(server_args)
    self.launch_desktop_session()
"""
NEW_LAUNCH = """    # --- %s: share the console's screen, do not start a virtual one ---
    # self._launch_server(server_args)
    # self.launch_desktop_session()
    self.child_env["DISPLAY"] = ":0"
    self.child_env["XAUTHORITY"] = "/run/user/%%d/gdm/Xauthority" %% os.getuid()
""" % MARKER

OLD_HOST = "      if desktop.server_proc is not None and desktop.host_proc is None:\n"
NEW_HOST = "      if desktop.host_proc is None:  # %s: no server of our own\n" % MARKER


def main():
    src = open(PATH).read()
    if MARKER in src:
        print("already applied")
        return 0
    if src.count(OLD_LAUNCH) != 1 or src.count(OLD_HOST) != 1:
        print("script shape changed; not touching it", file=sys.stderr)
        return 2
    backup = PATH + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(PATH, backup)
    out = src.replace(OLD_LAUNCH, NEW_LAUNCH).replace(OLD_HOST, NEW_HOST)
    with open(PATH, "w") as f:
        f.write(out)
    print("patched; original kept at", backup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
