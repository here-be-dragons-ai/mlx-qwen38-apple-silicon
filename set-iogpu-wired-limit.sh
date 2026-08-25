#!/bin/bash
# Setzt iogpu.wired_limit_mb auf den fuer DIESE Maschine passenden Wert.
#
# WARUM ES DIESES SKRIPT GIBT: der Wert haengt an der RAM-Groesse, und ein
# fest eingetragener Wert in einer plist ist auf jeder anderen Maschine falsch.
# Bis 2026-08-24 setzte com.local.iogpu-wired-limit.plist hart 26624 — den
# 32-GB-Wert. Auf der 48-GB-Testmaschine haette die so installierte plist beim
# naechsten Boot 14 GiB Working-Set verschenkt, und der Folgefehler ist STILL:
# start-mlx_qwen3.8.sh prueft `_WIRED_MB >= 40960` und faellt darunter auf
# PREFILL_STEP=512 zurueck; damit greift der NAX-Pfad aus PR #3842 (qL >= 1024)
# nicht mehr, ohne dass irgendwo eine Warnung erscheint.
#
# REGEL: RAM minus Reserve fuer macOS. Die Reserve zaehlt ABSOLUT, nicht
# relativ — auf groesseren Maschinen laeuft mehr nebenher:
#     <= 32 GB RAM  ->  6 GiB Reserve
#      > 32 GB RAM  ->  8 GiB Reserve
# Das reproduziert genau die drei belegten Werte:
#     32 GB -> 26624    48 GB -> 40960    64 GB -> 57344
#
# DIE OBERGRENZE IST NICHT VERHANDELBAR. 48 GB mit 45056 (44 GiB, also nur
# 4 GiB Reserve) ist am 2026-08-21 um 12:30 in eine Kernel-Panik gelaufen
# ("watchdog timeout: no checkins from watchdogd in 93 seconds"): 45,8 von
# 48 GiB wired, auslagerbarer Pool < 1 MB, der Pageout-Scanner bekam 252 von
# 3086 Seiten zurueck. Es gibt KEINE Vorwarnung — macOS' memoryPressure
# bewertet den Compressor, nicht Wired, und meldete durchgehend "false".
# Jetsam greift deshalb ebenfalls nicht. Dieses Clamp ist der einzige Schutz,
# und es gilt auch fuer manuell uebergebene Werte.
#
# Aufruf:
#   set-iogpu-wired-limit.sh              # berechnet und setzt
#   set-iogpu-wired-limit.sh 36864        # expliziter Wert (wird geclampt)
#   set-iogpu-wired-limit.sh --dry-run    # nur rechnen, nichts setzen
#
# Reihenfolge: Argument > /etc/iogpu-wired-limit.conf (eine Zahl) > berechnet.
set -euo pipefail

CONF=/etc/iogpu-wired-limit.conf
DRY=0
WANT=""

for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    [0-9]*)    WANT="$a" ;;
    *) echo "usage: $0 [MB] [--dry-run]" >&2; exit 2 ;;
  esac
done

RAM_MB=$(( $(/usr/sbin/sysctl -n hw.memsize) / 1048576 ))

# Reserve fuer macOS — absolut, s. Kommentar oben.
if (( RAM_MB <= 32768 )); then
  RESERVE_MB=6144
else
  RESERVE_MB=8192
fi
SAFE_MAX=$(( RAM_MB - RESERVE_MB ))

if [[ -z "$WANT" && -r "$CONF" ]]; then
  WANT=$(tr -dc '0-9' < "$CONF" || true)
fi
[[ -n "$WANT" ]] || WANT=$SAFE_MAX

# Clamp. Gilt ausdruecklich auch fuer explizite Werte: ein Tippfehler in der
# conf darf die Maschine nicht in die Panik fahren.
TARGET=$WANT
if (( TARGET > SAFE_MAX )); then
  echo "set-iogpu-wired-limit: ${WANT} MB laesst macOS nur $(( (RAM_MB - WANT) / 1024 )) GiB —" \
       "auf ${SAFE_MAX} MB geclampt (Reserve $(( RESERVE_MB / 1024 )) GiB)." >&2
  TARGET=$SAFE_MAX
fi
# Unter 2/3 des RAM bringt der Eingriff nichts: das ist der macOS-Default.
MIN_USEFUL=$(( RAM_MB * 2 / 3 ))
if (( TARGET < MIN_USEFUL )); then
  echo "set-iogpu-wired-limit: ${TARGET} MB liegt unter dem macOS-Default" \
       "(${MIN_USEFUL} MB) — das SENKT den Working-Set. Abbruch." >&2
  exit 1
fi

CUR=$(/usr/sbin/sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
echo "set-iogpu-wired-limit: RAM ${RAM_MB} MB · Reserve ${RESERVE_MB} MB · aktuell ${CUR} · Ziel ${TARGET}"

if (( DRY )); then
  exit 0
fi
if [[ "$CUR" == "$TARGET" ]]; then
  exit 0
fi
/usr/sbin/sysctl -w "iogpu.wired_limit_mb=${TARGET}"
