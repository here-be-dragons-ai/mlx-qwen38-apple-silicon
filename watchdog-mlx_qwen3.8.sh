#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Watchdog fuer start-mlx_qwen3.8.sh
#
# WARUM ES DAS GIBT — UND WARUM ES INZWISCHEN OPTIONAL IST
# Entstanden am 2026-08-22, als der Serverprozess ueber eine Sitzung immer mehr
# Speicher hielt: nach Modell+Drafter 15,96 GiB, nach dem ersten Generieren
# 25,42 GiB, im Leerlauf einer langen Sitzung 36,51 GiB — und der naechste
# groessere Prefill starb in "[METAL] Insufficient Memory".
#
# Der Sockel ist seit Patch 0015 weg (fusionierte Quantized-Linears, ~9 GiB).
# Einen Zuwachs PRO REQUEST gibt es nicht: bei konstanter Kontextlaenge steht
# der Idle-Wert ab dem zweiten Request still (8 x 13.460 Token, Delta 0,000).
# Die frueher vermuteten "~0,6 GiB pro Request" waren ein Messfehler — in jenen
# Reihen wuchs die Kontextlaenge mit, und das war der wachsende KV-Cache.
#
# Der Watchdog ist damit NICHT MEHR NOETIG, sondern ein Netz fuer
# unbeaufsichtigte Laeufe: eine einzelne Konversation kann immer noch lang
# genug werden, um den Working-Set zu fuellen.
#
# BENUTZUNG
#   ./watchdog-mlx_qwen3.8.sh              # statt ./start-mlx_qwen3.8.sh
#   WATCHDOG_PCT=85 ./watchdog-mlx_qwen3.8.sh
# Alle Variablen des Start-Skripts werden durchgereicht:
#   ENABLE_APC=0 PROFILE=roomy ./watchdog-mlx_qwen3.8.sh
#
# Ctrl-C beendet Watchdog UND Server.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

HERE="${0:A:h}"
START_SCRIPT="${START_SCRIPT:-$HERE/start-mlx_qwen3.8.sh}"

STATE_DIR="${STATE_DIR:-$HOME/.mlx-qwen38}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"

# Schwelle in Prozent des Metal-Working-Sets. 90 laesst genug Luft fuer einen
# laufenden Prefill; 95 ist erfahrungsgemaess zu spaet (bei 95 % starb der
# 29.632-Token-Prefill am 2026-08-22 nach 34 %).
WATCHDOG_PCT="${WATCHDOG_PCT:-90}"
# So viele Messungen in Folge ueber der Schwelle, bevor neu gestartet wird.
# Der Sampler misst auch waehrend eines Prefills, und dort ist ein kurzer
# Ausschlag normal — ohne Streak wuerde der Watchdog flattern.
WATCHDOG_STREAK="${WATCHDOG_STREAK:-3}"
WATCHDOG_POLL="${WATCHDOG_POLL:-10}"
# Nach so vielen Sekunden Warten auf Leerlauf wird trotzdem neu gestartet. Ein
# Server, der dauerhaft ueber der Schwelle UND dauerhaft beschaeftigt ist, wird
# den naechsten grossen Prefill ohnehin nicht ueberleben.
WATCHDOG_MAX_WAIT="${WATCHDOG_MAX_WAIT:-180}"

SERVER_PID=""

log() { print -r -- "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] $*"; }

cleanup() {
  log "beende Server (PID ${SERVER_PID:-?})"
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  pkill -f "mlx_vlm.server" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

start_server() {
  # MEM_PROBE_INTERVAL darf nicht 0 sein — der Watchdog liest genau diese Zeilen.
  if [[ "${MEM_PROBE_INTERVAL:-5}" == "0" ]]; then
    log "MEM_PROBE_INTERVAL=0 wuerde den Sampler abschalten; setze 5."
    export MEM_PROBE_INTERVAL=5
  fi
  "$START_SCRIPT" &
  SERVER_PID=$!
  log "Server gestartet (PID $SERVER_PID), Schwelle ${WATCHDOG_PCT}%"
}

# Letzter Prozentwert aus den mem-Zeilen des Sampler-Threads.
current_pct() {
  local line
  line=$(grep "mem active=" "$LOG_FILE" 2>/dev/null | tail -1) || return 1
  [[ -z "$line" ]] && return 1
  print -r -- "$line" | sed -n 's/.*(\([0-9]*\)% von.*/\1/p'
}

# Leerlauf: die zuletzt protokollierte in_flight-Zahl ist 0. Der Server schreibt
# sie bei jedem Request-Ende mit.
is_idle() {
  local last
  last=$(grep -o "in_flight=[0-9]*" "$LOG_FILE" 2>/dev/null | tail -1) || return 1
  [[ "$last" == "in_flight=0" ]]
}

restart_server() {
  # ACHTUNG bei Arithmetik unter `set -e`: (( x++ )) liefert den ALTEN Wert als
  # Exit-Status, ist bei x=0 also "falsch" und beendet das Skript. Deshalb
  # ueberall x=$(( x + 1 )) statt (( x++ )), und bei Tests ein || true.
  local waited=0
  while ! is_idle; do
    if (( waited >= WATCHDOG_MAX_WAIT )); then
      log "seit ${waited}s kein Leerlauf — starte trotzdem neu (ein Request wird abgebrochen)"
      break
    fi
    (( waited == 0 )) && log "warte auf Leerlauf, bevor neu gestartet wird" || true
    sleep 5
    waited=$(( waited + 5 ))
  done

  log "Neustart"
  kill "$SERVER_PID" 2>/dev/null || true
  # Dem uvicorn-Shutdown Zeit geben, danach hart.
  local n=0
  while kill -0 "$SERVER_PID" 2>/dev/null && (( n < 15 )); do sleep 1; n=$(( n + 1 )); done
  pkill -f "mlx_vlm.server" 2>/dev/null || true
  sleep 3
  start_server
}

start_server
streak=0

while true; do
  sleep "$WATCHDOG_POLL"

  # Server unerwartet weg (Absturz, OOM-Kill)? Dann neu hoch.
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Serverprozess ist weg — starte neu"
    sleep 3
    start_server
    streak=0
    continue
  fi

  pct=$(current_pct) || continue
  [[ -z "$pct" ]] && continue

  if (( pct >= WATCHDOG_PCT )); then
    streak=$(( streak + 1 ))
    log "${pct}% >= ${WATCHDOG_PCT}%  (${streak}/${WATCHDOG_STREAK})"
    if (( streak >= WATCHDOG_STREAK )); then
      restart_server
      streak=0
    fi
  else
    (( streak > 0 )) && log "wieder unter der Schwelle (${pct}%)" || true
    streak=0
  fi
done
