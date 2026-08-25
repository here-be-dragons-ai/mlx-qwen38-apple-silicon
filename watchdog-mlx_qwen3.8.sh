#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Watchdog for start-mlx_qwen3.8.sh
#
# WHY IT EXISTS -- AND WHY IT IS OPTIONAL BY NOW
# Written on 2026-08-22, when the server process held more and more memory over
# a session: 15.96 GiB after model+drafter, 25.42 GiB after the first
# generation, 36.51 GiB while idle in a long session -- and the next larger
# prefill died in "[METAL] Insufficient Memory".
#
# That floor is gone since patch 0015 (fused quantized linears, ~9 GiB). There
# is no growth PER REQUEST: at constant context length the idle value stands
# still from the second request onwards (8 x 13,460 tokens, delta 0.000). The
# "~0.6 GiB per request" suspected earlier was a measurement error -- in those
# series the context length grew along, and that was the growing KV cache.
#
# The watchdog is therefore NO LONGER NECESSARY, but a net for unattended runs:
# a single conversation can still grow long enough to fill the working set.
#
# USAGE
#   ./watchdog-mlx_qwen3.8.sh              # instead of ./start-mlx_qwen3.8.sh
#   WATCHDOG_PCT=85 ./watchdog-mlx_qwen3.8.sh
# All variables of the start script are passed through:
#   ENABLE_APC=0 PROFILE=roomy ./watchdog-mlx_qwen3.8.sh
#
# Ctrl-C terminates watchdog AND server.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

HERE="${0:A:h}"
START_SCRIPT="${START_SCRIPT:-$HERE/start-mlx_qwen3.8.sh}"

STATE_DIR="${STATE_DIR:-$HOME/.mlx-qwen38}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"

# Threshold in percent of the Metal working set. 90 leaves enough room for a
# running prefill; experience says 95 is too late (at 95% the 29,632-token
# prefill died at 34% on 2026-08-22).
WATCHDOG_PCT="${WATCHDOG_PCT:-90}"
# This many consecutive measurements above the threshold before restarting. The
# sampler also measures during a prefill, where a brief spike is normal --
# without a streak the watchdog would flap.
WATCHDOG_STREAK="${WATCHDOG_STREAK:-3}"
WATCHDOG_POLL="${WATCHDOG_POLL:-10}"
# After this many seconds of waiting for idle, restart anyway. A server that is
# permanently above the threshold AND permanently busy will not survive the next
# large prefill regardless.
WATCHDOG_MAX_WAIT="${WATCHDOG_MAX_WAIT:-180}"

SERVER_PID=""

log() { print -r -- "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] $*"; }

cleanup() {
  log "stopping server (PID ${SERVER_PID:-?})"
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  pkill -f "mlx_vlm.server" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

start_server() {
  # MEM_PROBE_INTERVAL must not be 0 -- the watchdog reads exactly those lines.
  if [[ "${MEM_PROBE_INTERVAL:-5}" == "0" ]]; then
    log "MEM_PROBE_INTERVAL=0 would disable the sampler; setting 5."
    export MEM_PROBE_INTERVAL=5
  fi
  "$START_SCRIPT" &
  SERVER_PID=$!
  log "server started (PID $SERVER_PID), threshold ${WATCHDOG_PCT}%"
}

# Last percentage from the mem lines of the sampler thread.
#
# The pattern deliberately matches only "(NN%" up to the closing parenthesis and
# NOT the wording that follows. That wording lives in the Python probe inside
# start-mlx_qwen3.8.sh, and coupling the watchdog to its exact phrasing means a
# reworded log line silently disables the watchdog -- it would find no value,
# `continue` on every poll and never restart anything.
current_pct() {
  local line
  line=$(grep "mem active=" "$LOG_FILE" 2>/dev/null | tail -1) || return 1
  [[ -z "$line" ]] && return 1
  print -r -- "$line" | sed -n 's/.*(\([0-9]*\)%[^)]*).*/\1/p'
}

# Idle: the last logged in_flight count is 0. The server writes it on every
# request completion.
is_idle() {
  local last
  last=$(grep -o "in_flight=[0-9]*" "$LOG_FILE" 2>/dev/null | tail -1) || return 1
  [[ "$last" == "in_flight=0" ]]
}

restart_server() {
  # CAREFUL with arithmetic under `set -e`: (( x++ )) yields the OLD value as
  # exit status, so at x=0 it is "false" and terminates the script. Hence
  # x=$(( x + 1 )) everywhere instead of (( x++ )), and || true on tests.
  local waited=0
  while ! is_idle; do
    if (( waited >= WATCHDOG_MAX_WAIT )); then
      log "no idle for ${waited}s -- restarting anyway (one request will be aborted)"
      break
    fi
    (( waited == 0 )) && log "waiting for idle before restarting" || true
    sleep 5
    waited=$(( waited + 5 ))
  done

  log "restart"
  kill "$SERVER_PID" 2>/dev/null || true
  # Give the uvicorn shutdown some time, then go hard.
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

  # Server unexpectedly gone (crash, OOM kill)? Bring it back up.
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "server process is gone -- restarting"
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
    (( streak > 0 )) && log "back below the threshold (${pct}%)" || true
    streak=0
  fi
done
