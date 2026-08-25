#!/bin/bash
# Installs set-iogpu-wired-limit.sh + a LaunchDaemon so the GPU wired limit
# survives every reboot. Idempotent: safe to run repeatedly, replaces an
# existing installation.
#
# WHY A SCRIPT: the individual steps are four long sudo lines with continuation
# characters. Pasted into a terminal that breaks in the wrong place, zsh then
# executes the target directory as a command ("permission denied:
# /usr/local/libexec/") and `launchctl` runs on without sudo ("Warning:
# Expecting a LaunchAgents path ... Load failed: 5"). That is exactly how the
# installation half-completed on 2026-08-24: directory there, script and plist
# not. A single call cannot break that way.
#
#   sudo ./install-wired-limit-daemon.sh             # install
#   sudo ./install-wired-limit-daemon.sh --uninstall # remove
set -euo pipefail

LABEL=com.local.iogpu-wired-limit
PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/${LABEL}.plist"
HELPER_SRC="$(cd "$(dirname "$0")" && pwd)/set-iogpu-wired-limit.sh"
PLIST_DST="/Library/LaunchDaemons/${LABEL}.plist"
HELPER_DST="/usr/local/libexec/set-iogpu-wired-limit.sh"

if [[ "$(id -u)" != "0" ]]; then
  echo "This script needs root: sudo $0 $*" >&2
  exit 1
fi

# Tear down an existing instance -- otherwise bootstrap fails with
# "Bootstrap failed: 37: Operation already in progress".
unload_daemon() {
  launchctl bootout "system/${LABEL}" 2>/dev/null \
    || launchctl unload -w "$PLIST_DST" 2>/dev/null \
    || true
}

if [[ "${1:-}" == "--uninstall" ]]; then
  unload_daemon
  rm -f "$PLIST_DST" "$HELPER_DST"
  echo "Removed. iogpu.wired_limit_mb stays at $(sysctl -n iogpu.wired_limit_mb) until reboot."
  exit 0
fi

for f in "$PLIST_SRC" "$HELPER_SRC"; do
  [[ -r "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done

echo "before: iogpu.wired_limit_mb = $(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo '?')"

# root:wheel and 755 are mandatory, not cosmetic: the daemon runs as root, and a
# user-writable script in that location would be a privilege escalation.
install -d -o root -g wheel -m 755 /usr/local/libexec
install -o root -g wheel -m 755 "$HELPER_SRC" "$HELPER_DST"
install -o root -g wheel -m 644 "$PLIST_SRC"  "$PLIST_DST"

unload_daemon
# bootstrap is the modern form; load -w as a fallback for older macOS.
launchctl bootstrap system "$PLIST_DST" 2>/dev/null \
  || launchctl load -w "$PLIST_DST"

# RunAtLoad takes effect immediately, so the value must already be in place.
echo "after:  iogpu.wired_limit_mb = $(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo '?')"
echo
echo "Loaded:"
launchctl print "system/${LABEL}" 2>/dev/null | grep -E "state|program|last exit" | sed 's/^/  /' \
  || echo "  (launchctl print unavailable -- check with 'sudo launchctl list | grep iogpu')"
echo
echo "Log: /var/log/iogpu-wired-limit.log"
echo "Test without rebooting: sudo $HELPER_DST --dry-run"
