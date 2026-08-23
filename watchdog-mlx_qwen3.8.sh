#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Watchdog fuer start-mlx_qwen3.8.sh
#
# WARUM ES DAS GIBT
# Der Serverprozess sammelt ueber eine lange Agent-Sitzung lebenden Speicher an,
# der nie zurueckgegeben wird. Gemessen am 2026-08-21/22:
#   nach Modell+Drafter, idle                    active = 15,96 GiB
#   nach dem ersten Generieren (auch 16 Token!)  active = 25,42 GiB  (fixer Sockel)
#   danach ~0,6 GiB pro Request, kumulativ       active = 36,51 GiB im Leerlauf
# Bei 40 GiB Working-Set bleiben dann ~3,5 GiB, und der naechste groessere
# Prefill stirbt in "[METAL] Insufficient Memory". Ein Neustart setzt
# nachweislich auf die 15,96-GiB-Baseline zurueck.
#
# Das hier ist SYMPTOMBEKAEMPFUNG. Die Ursachen sind nicht verstanden (Sockel
# haengt an --draft-block-size >= 2, der Kriechgang ist offen). Bis dahin ist ein
# rechtzeitiger Neustart billiger als ein Abbruch mitten in einer Antwort.
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
