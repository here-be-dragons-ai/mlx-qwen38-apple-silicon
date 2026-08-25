#!/bin/bash
# Sets iogpu.wired_limit_mb to the value appropriate for THIS machine.
#
# WHY THIS SCRIPT EXISTS: the value depends on the amount of RAM, and a value
# hardcoded into a plist is wrong on every other machine. Until 2026-08-24
# com.local.iogpu-wired-limit.plist set a fixed 26624 -- the 32 GB value. On the
# 48 GB test machine that plist, once installed, would have thrown away 14 GiB of
# working set at the next boot, and the follow-on error is SILENT:
# start-mlx_qwen3.8.sh checks `_WIRED_MB >= 40960` and falls back to
# PREFILL_STEP=512 below that, which stops the NAX path from PR #3842
# (qL >= 1024) from engaging at all, without any warning appearing anywhere.
#
# RULE: RAM minus a reserve for macOS. The reserve is ABSOLUTE, not relative --
# more runs alongside on larger machines:
#     <= 32 GB RAM  ->  6 GiB reserve
#      > 32 GB RAM  ->  8 GiB reserve
# That reproduces exactly the three documented values:
#     32 GB -> 26624    48 GB -> 40960    64 GB -> 57344
#
# THE UPPER BOUND IS NOT NEGOTIABLE. 48 GB with 45056 (44 GiB, so only 4 GiB of
# reserve) ran into a kernel panic on 2026-08-21 at 12:30 ("watchdog timeout: no
# checkins from watchdogd in 93 seconds"): 45.8 of 48 GiB wired, pageable pool
# below 1 MB, and the pageout scanner got back 252 of 3086 pages. There is NO
# warning -- macOS' memoryPressure evaluates the compressor, not wired memory,
# and reported "false" throughout. Jetsam does not step in either. This clamp is
# the only protection, and it applies to hand-passed values as well.
#
# Usage:
#   set-iogpu-wired-limit.sh              # compute and set
#   set-iogpu-wired-limit.sh 36864        # explicit value (still clamped)
#   set-iogpu-wired-limit.sh --dry-run    # compute only, set nothing
#
# Precedence: argument > /etc/iogpu-wired-limit.conf (a single number) > computed.
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

# Reserve for macOS -- absolute, see comment above.
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

# Clamp. Explicitly applies to explicit values too: a typo in the conf file must
# not be able to drive the machine into a panic.
TARGET=$WANT
if (( TARGET > SAFE_MAX )); then
  echo "set-iogpu-wired-limit: ${WANT} MB leaves macOS only $(( (RAM_MB - WANT) / 1024 )) GiB --" \
       "clamped to ${SAFE_MAX} MB (reserve $(( RESERVE_MB / 1024 )) GiB)." >&2
  TARGET=$SAFE_MAX
fi
# Below 2/3 of RAM the intervention achieves nothing: that is the macOS default.
MIN_USEFUL=$(( RAM_MB * 2 / 3 ))
if (( TARGET < MIN_USEFUL )); then
  echo "set-iogpu-wired-limit: ${TARGET} MB is below the macOS default" \
       "(${MIN_USEFUL} MB) -- that would LOWER the working set. Aborting." >&2
  exit 1
fi

CUR=$(/usr/sbin/sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
echo "set-iogpu-wired-limit: RAM ${RAM_MB} MB · reserve ${RESERVE_MB} MB · current ${CUR} · target ${TARGET}"

if (( DRY )); then
  exit 0
fi
if [[ "$CUR" == "$TARGET" ]]; then
  exit 0
fi
/usr/sbin/sysctl -w "iogpu.wired_limit_mb=${TARGET}"
