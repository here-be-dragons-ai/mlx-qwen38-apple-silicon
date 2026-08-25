#!/bin/bash
# Installiert set-iogpu-wired-limit.sh + LaunchDaemon, sodass das GPU-Wired-Limit
# jeden Reboot ueberlebt. Idempotent: mehrfach aufrufbar, ersetzt eine bestehende
# Installation.
#
# WARUM ALS SKRIPT: die Einzelschritte sind vier lange sudo-Zeilen mit
# Fortsetzungszeichen. Beim Kopieren in ein Terminal bricht das an der falschen
# Stelle um, zsh fuehrt dann das Zielverzeichnis als Kommando aus
# ("permission denied: /usr/local/libexec/") und `launchctl` laeuft ohne sudo
# weiter ("Warning: Expecting a LaunchAgents path ... Load failed: 5"). Genau so
# ist die Installation am 2026-08-24 halb durchgelaufen: Verzeichnis da, Skript
# und plist nicht. Ein Aufruf ist nicht zu zerbrechen.
#
#   sudo ./install-wired-limit-daemon.sh            # installieren
#   sudo ./install-wired-limit-daemon.sh --uninstall # entfernen
set -euo pipefail

LABEL=com.local.iogpu-wired-limit
PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/${LABEL}.plist"
HELPER_SRC="$(cd "$(dirname "$0")" && pwd)/set-iogpu-wired-limit.sh"
PLIST_DST="/Library/LaunchDaemons/${LABEL}.plist"
HELPER_DST="/usr/local/libexec/set-iogpu-wired-limit.sh"

if [[ "$(id -u)" != "0" ]]; then
  echo "Dieses Skript braucht root: sudo $0 $*" >&2
  exit 1
fi

# Bestehende Instanz abraeumen — sonst scheitert bootstrap mit
# "Bootstrap failed: 37: Operation already in progress".
unload_daemon() {
  launchctl bootout "system/${LABEL}" 2>/dev/null \
    || launchctl unload -w "$PLIST_DST" 2>/dev/null \
    || true
}

if [[ "${1:-}" == "--uninstall" ]]; then
  unload_daemon
  rm -f "$PLIST_DST" "$HELPER_DST"
  echo "Entfernt. iogpu.wired_limit_mb bleibt bis zum Reboot auf $(sysctl -n iogpu.wired_limit_mb)."
  exit 0
fi

for f in "$PLIST_SRC" "$HELPER_SRC"; do
  [[ -r "$f" ]] || { echo "FEHLT: $f" >&2; exit 1; }
done

echo "vorher:  iogpu.wired_limit_mb = $(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo '?')"

# root:wheel und 755 sind Pflicht, nicht Kosmetik: der Daemon laeuft als root,
# ein fuer den Benutzer schreibbares Skript an dieser Stelle waere eine
# Rechteausweitung.
install -d -o root -g wheel -m 755 /usr/local/libexec
install -o root -g wheel -m 755 "$HELPER_SRC" "$HELPER_DST"
install -o root -g wheel -m 644 "$PLIST_SRC"  "$PLIST_DST"

unload_daemon
# bootstrap ist die moderne Form; load -w als Rueckfall fuer aeltere macOS.
launchctl bootstrap system "$PLIST_DST" 2>/dev/null \
  || launchctl load -w "$PLIST_DST"

# RunAtLoad greift sofort, der Wert muss also jetzt schon stehen.
echo "nachher: iogpu.wired_limit_mb = $(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo '?')"
echo
echo "Geladen:"
launchctl print "system/${LABEL}" 2>/dev/null | grep -E "state|program|last exit" | sed 's/^/  /' \
  || echo "  (launchctl print nicht verfuegbar — 'sudo launchctl list | grep iogpu' pruefen)"
echo
echo "Log: /var/log/iogpu-wired-limit.log"
echo "Test ohne Reboot: sudo $HELPER_DST --dry-run"
