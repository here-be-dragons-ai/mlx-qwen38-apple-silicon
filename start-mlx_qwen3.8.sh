#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm server start script  -  Qwen3.8 27B (DENSE, MLX 4bit)
# TARGET HARDWARE:  Apple Silicon, 32 GB unified memory and up
#
# The memory-relevant parameters come from PROFILE (see below):
#     lean      32 GB without sudo  (working set 21.3 GiB)  peak ~18.8 GiB
#     balanced  32 GB with sysctl   (working set 26 GiB)    peak ~25.0 GiB
#     roomy     48 GB               (working set 40 GiB)    peak ~33 GiB (est.)
#     auto      (default) picks based on the working set
#
# All MEASURED VALUES in this script come from the 48 GB machine (M5 Pro,
# mlx-vlm 0.6.13/0.6.15) on which the "roomy" profile ran for weeks. What is
# carried over to 32 GB are the hardware-independent findings (MTP spec-dec is
# worth it, APC is worth it, KV windowing hallucinates); the MEMORY parameters
# are computed.
#
# ── DOES THE MODEL FIT IN 32 GB? YES -- but context is the bottleneck. ───────
#
#   weights         14.95 GiB (3 shards, 4bit affine, group_size 64)
#   MTP drafter      0.23 GiB (mlx-community/Qwen3.8-27B-MTP-4bit)
#   ──────────────────────────
#   fixed          ~15.2 GiB   <- fits even in the macOS default (see below)
#
#   On top of that, PER SEQUENCE AND PER APC SNAPSHOT:
#     KV cache      64 KiB/token  (16 full-attn layers x 4 KV heads x 512 x 2 B)
#                                  with KV_BITS=8: 32 KiB, KV_BITS=4: 16 KiB
#     GDN state    ~152 MiB fixed  (48 linear layers x 48 v-heads x 128 x 128 x 4 B,
#                                  mamba_ssm_dtype=float32; length-independent)
#
#   Metal's max_recommended_working_set_size is 2/3 of RAM by default on Macs
#   <= 36 GB  ->  32 GB = 21.33 GiB. After the weights, only ~4.6 GiB of that is
#   left for KV + snapshots + activations. Hence:
#
#     WITHOUT sysctl (21.3 GiB budget):  ~23k tokens context  -> client ctx 24576
#     WITHOUT sysctl, but KV_BITS=8   :  ~46k tokens context  -> client ctx 32768
#     WITH    sysctl 26624 MB (26 GiB):  ~48k tokens context  -> client ctx 49152
#     WITH    sysctl + KV_BITS=8      :  ~97k tokens context  -> client ctx 65536
#
#   The script COMPUTES this budget at start from the real values of the running
#   machine and prints it (the "memory budget" block) -- the numbers above are
#   only the expectation for a bare 32 GB machine.
#
#   Setting the wired limit (not persistent, needs sudo):
#       sudo sysctl -w iogpu.wired_limit_mb=26624
#   Persistent: install-wired-limit-daemon.sh from this directory, see README.
#   NOT higher than 26624 on 32 GB -- macOS itself needs ~5-6 GB below that;
#   setting the limit too high trades a Metal OOM for a kernel panic/beachball.
#
# ── EXPECTED SPEED (ESTIMATED, not measured on an M5 base) ──────────────────
#   The model is DENSE: every decode step reads ~15 GiB -> purely a bandwidth
#   question. The M5 base has ~153 GB/s, the measured M5 Pro considerably more.
#   Scaled from the 48 GB measurements (17.5-18.4 t/s decode):
#       decode raw             ~8-10 t/s
#       decode with MTP spec-dec ~13-20 t/s on tool calls/JSON
#                              (acceptance measured 90-93% there, 42% on prose)
#       prefill                ~180-250 t/s (M5 Pro: 420-470 t/s, GPU cores/2)
#   Consequence for agent operation: a COLD 30k prefill takes ~2-3 minutes.
#   Which is exactly why APC + the SSD tier matter even more here than on the
#   48 GB box (measured there: 89,630 ms -> 350 ms at 36k tokens, factor 256).
#
# Prerequisite: install-prereqs.sh has run, port 8888 is free.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths (all overridable via env; no hardcoded user name) ───────────────────
VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"
MODELS_ROOT="${MLX_MODELS:-$HOME/src/mlx/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/Qwen3.8-27B-MLX-4bit}"
# Runtime data: log and SSD prefix cache. To put them elsewhere, set STATE_DIR --
# or LOG_FILE / APC_DISK individually.
STATE_DIR="${STATE_DIR:-$HOME/.mlx-qwen38}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"
# Do NOT use $HOST: zsh occupies that parameter itself with the host name, so a
# "${HOST:-127.0.0.1}" would bind the server to the LAN address.
BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8888}"

# ── Model alias ───────────────────────────────────────────────────────────────
# mlx-vlm has NO --alias: the "model" string from the request is the LOAD PATH.
# If it differs from the preload path, the server discards the loaded model and
# starts a snapshot_download() against HuggingFace (-> 401, even though it is
# local). Solution: a symlink with exactly the alias name next to the model, and
# preload the server with the RELATIVE name (cd to MODELS_ROOT at the end).
MODEL_ALIAS="${MODEL_ALIAS:-Qwen3.8-27B-local}"

# ── Profiles ──────────────────────────────────────────────────────────────────
# A profile only sets DEFAULTS -- every individually set env variable still
# wins, e.g.  PROFILE=lean APC_ENTRIES=2 ./start-mlx_qwen3.8.sh
#
#   lean      Minimal RAM. Peak ~18.8 GiB at a 29k peak prompt -> fits under the
#             macOS default working set of 21.33 GiB, i.e. WITHOUT sudo.
#             Computed saving against balanced (43k prompt):
#               APC_ENTRIES 2->1 (3->2 copies)  -2.8 GiB   cheapest lever: the
#                  SSD tier remains, an evicted snapshot is back in 350 ms
#                  instead of a 90 s full prefill
#               KV_BITS=8 (64->32 KiB/token)    -4.2 GiB   throughput cost on the
#                  MLX path NOT measured -- on llama.cpp, KV quant at a
#                  comparable place cost up to 8x prefill. A/B before production.
#               PREFILL_STEP 1024->512          ~-0.2 GiB  transient peak
#
#   balanced  32 GB with a raised wired limit (26624 MB). Peak ~25 GiB at a 43k
#             peak prompt, KV unquantized -- no unmeasured trade-off.
#             (Alias: "default", the historical name.)
#
#   roomy     48 GB. The original M5 Pro setup, run for weeks:
#             PREFILL_STEP 2048, VISION_CACHE 20, APC_DISK_MAX_GB 80.
#             -> needs iogpu.wired_limit_mb=40960. NOT 45056: that is 44 GiB out
#               of 48 GiB, leaving macOS ~2 GiB, and that is exactly what drove
#               the machine into a kernel panic on 2026-08-21.
#             APC_ENTRIES was 4 here until 2026-08-20, then 1, and is 3 since
#             2026-08-24 (see the block above the profile). A snapshot is as
#             large as the prompt (64 KiB/token), so 6.5 GiB at 104k tokens --
#             FOUR of those are 26 GiB and blow the budget as soon as the context
#             is really used. The current 3 rests on all clients running
#             context_length 65536; see "WHY 3" below.
#
# Each profile has a matching client-side context_length -- the banner prints it,
# and further down the script warns when the config.yaml exceeds it.
PROFILE="${PROFILE:-auto}"
# auto: picks based on the Metal working set. It is estimated via sysctl here
# rather than mx.device_info(), because the profile defaults are needed BEFORE
# the venv Python runs (loading the model takes a while). Rule:
# iogpu.wired_limit_mb wins when set; otherwise the macOS default (2/3 of RAM at
# <= 36 GB, else 3/4). The exact number from Metal appears in the banner later --
# when the two differ, the banner is authoritative.
# The working set is ALWAYS computed, not only at PROFILE=auto: roomy decides
# below, based on _WIRED_MB, how many APC snapshots fit the budget.
_RAM_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))
_WIRED_MB=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if [[ "${_WIRED_MB:-0}" -gt 0 ]]; then
  _WS_MB=$_WIRED_MB
elif [[ "$_RAM_MB" -le 36864 ]]; then
  _WS_MB=$(( _RAM_MB * 2 / 3 ))
else
  _WS_MB=$(( _RAM_MB * 3 / 4 ))
fi

if [[ "$PROFILE" == "auto" ]]; then
  if   [[ "$_WS_MB" -ge 30720 ]]; then PROFILE=roomy      # >= 30 GiB
  elif [[ "$_WS_MB" -ge 24576 ]]; then PROFILE=balanced   # >= 24 GiB
  else                                 PROFILE=lean
  fi
  _AUTO_NOTE=" (auto, working set ~$(( _WS_MB / 1024 )) GiB)"
else
  _AUTO_NOTE=""
fi

case "$PROFILE" in
  # VISION_CACHE is 1 here and NOT 0: VisionFeatureCache.put() checks
  # `len(cache) >= max_size` and then calls popitem() -- at max_size=0 that is on
  # an empty OrderedDict, which kills the first image request with a KeyError.
  # 1 is the real minimum (vision_cache.py:60ff).
  lean)
    _APC_ENTRIES=1; _KV_BITS="8"; _PREFILL=512;  _VISION=1;  _APC_MAXGB=40; _APC_MINFREE=4.0; _CTX_HINT=32768 ;;
  balanced|default)
    PROFILE=balanced
    _APC_ENTRIES=2; _KV_BITS="";  _PREFILL=1024; _VISION=4;  _APC_MAXGB=40; _APC_MINFREE=4.0; _CTX_HINT=49152 ;;
  # ── HISTORY OF THE roomy NUMBERS ───────────────────────────────────────────
  # Kept because it explains why the current values are what they are. The
  # figures below are UNFUSED and predate the mlx source build; with patch 0013
  # active the prefill transient disappears and the budgets are considerably
  # higher. The banner line is authoritative, not this comment.
  #
  # _CTX_HINT was 98304 for a long time, with the argument that 131072 would only
  # fit WITH wired_limit set (and with the historical 45056). At the macOS
  # default (48 GB -> 37.4 GiB working set) it is ~107k, so 131072 would be
  # overbooked, while 98304 carried in both cases.
  # That argument is obsolete since 2026-08-24: _CTX_HINT is not a server limit,
  # it is the input to the overbooking guard, and at 98304 that guard capped
  # APC_ENTRIES to 1 no matter what was configured. See "WHY 3" below.
  #
  # From 2026-08-20 to 2026-08-21, APC_ENTRIES was additionally coupled to the
  # wired limit -- three snapshots instead of two as soon as the limit was set.
  # That coupling is REVERTED. It rested on the budget calculation that the
  # memory sampler disproved, and it had an unpleasant side effect: raising the
  # limit gained 2.5 GiB of ceiling but doubled the snapshot count at the same
  # time, a net regression.
  # What the coupling was based on stays valid as a measurement, and it is the
  # reason a third slot exists at all: on 2026-08-20 the production log showed
  # 168 of 549 prefills running cold (30.6%), 45 of them above 8k tokens = 2415 s
  # of pure prefill time. The expensive cases are NOT new conversations -- in the
  # window 10:03-10:22 there was no server restart, yet three requests fell to
  # cached_tokens=0 (50.4 s / 56.8 s / 70.7 s) between warm turns of the same
  # size. That is eviction: more simultaneously active conversations than
  # snapshot slots.
  #
  # PREFILL_STEP also hangs on the wired limit, for a different reason than
  # APC_ENTRIES: head_dim is 256, and mlx's default dispatch permits fused full
  # attention (up to 0.32.1) only for 64/80/128. The 16 full-attn layers
  # therefore each materialise a score tensor of n_heads x qL x kL x 2 B -- at
  # chunk 2048 and 38k context that is 2.8 GiB PER LAYER, and delayed evaluation
  # keeps several of them alive at once.
  # MEASURED on a 37.4 GiB working set, cold prefill, tier cleared beforehand:
  #   chunk 2048               up to ~30k, then [METAL] Insufficient Memory
  #   chunk 2048 + KV_BITS=8   up to ~30k  -- halving KV changes NOTHING, the
  #                                          limit is the score tensor
  #   chunk  512               up to ~38k  ok
  # So 512 without the limit. With the limit it stays 2048 -- there is room
  # there, and larger chunks are somewhat more efficient during prefill.
  # THIS APPLIES ONLY UNFUSED. Since the mlx source build (0.32.2.dev,
  # 2026-08-21) patch 0013 is live, the score tensor disappears, and the 512 is
  # raised to 1024 further down -- the NAX path from PR #3842 requires
  # qL >= 1024. These lines stay because they apply again without a source build.
  # ── APC_ENTRIES: 3, and the "factor 7" is resolved ─────────────────────────
  # This said 3 until 2026-08-21 ("as soon as the limit is set"), then a
  # conservative 1. The justification for the 1 was the memory sampler of
  # 2026-08-21 (40 GiB working set, APC_ENTRIES=3, KV_BITS=8):
  #   after model+drafter, idle    active = 15.96 GiB
  #   after 1 request of 13,112 tk active = 27.35 GiB   <- +11.4 instead of 1.6
  # That "factor 7" IS EXPLAINED and was not a snapshot problem: 9.46 GiB of it
  # is the fixed floor from _fused_quantized_linears() (patch 0015, see
  # docs/memory.md "The fixed floor"). 11.4 - 9.46 = 1.94 GiB against a computed 1.2-1.6 -- that
  # matches. The floor has been switched off since patch 0015, `active` after the
  # first request is 16.70 instead of 25.42 GiB, and there is no creep per
  # request (counter-test at constant length, commit 72ba66c).
  # So the calculation no longer lies, and the 1 costs measurable prefill.
  #
  # WHAT THE 1 COSTS, from server.log (810 prefills, 118 server runs):
  #   warm 503 (62.1%, 9.71 M tokens reused) · cold 307 (37.9%)
  #   cold <8k    184   602 s   (irrelevant)
  #   cold 8-20k   40  1128 s
  #   cold >20k    83  5331 s
  #   ------------------------------------------  cold >=8k = 107.7 min
  # Those 107.7 min split into TWO different faults, and only the first hangs on
  # APC_ENTRIES:
  #   72.7 min (81 prefills) follow an earlier large request within the same
  #            server run -- that is eviction on too few snapshot slots
  #            (in_flight=2 appears 5x in the log). Upper bound: some of them are
  #            genuinely new conversations that no slot would save.
  #   34.9 min (42 prefills) are the first large request of their run. Those the
  #            DISK tier should have caught -- and that is exactly what the
  #            namespace garbage plus the wrong cap arithmetic destroyed
  #            (19 evictions in the log, see APC GC further down). Different fix.
  #
  # WHY 3, AND WHY _CTX_HINT HAS TO BE 65536 FOR IT.
  # On 2026-08-24 this first said 2, computed against _CTX_HINT=98304. That
  # calculation was not wrong -- it answered the wrong question. Because:
  #
  #   mlx-vlm has NO -c flag. The server context comes from config.json
  #   (262144); the effective ceiling is ONLY the client's contextWindow (see the
  #   "client reconciliation" block below). _CTX_HINT is a recommendation for the
  #   banner and the input to the overbooking guard -- not a server limit.
  #
  # All three pi instances (~/.pi/agent, ~/.pi/ea, ~/.pi/code) run contextWindow
  # 65536. _CTX_HINT=98304 therefore described a client that does not exist, and
  # reserved memory for prompts nobody sends.
  #
  # THE CONSEQUENCE WAS SILENT AND EXPENSIVE. The overbooking guard further down
  # reduces entries while ctx_hint > budget * 0.8. Computed with this script's
  # own budget block (40.0 GiB working set, 16.0 GiB weights, patch 0013 active),
  # field entries_used:
  #   _CTX_HINT     set    entries_used   copies   budget
  #     98304        2          1 (!)        2     182133
  #     98304        3          1 (!)        2     182133
  #     65536        2          2            3     120654
  #     65536        3          3            4      89914
  # At 98304 EVERY setting ends up at 1 -- so the 2 set earlier that day never
  # took effect, and the gain it was about (72.7 min of cold prefill, see above)
  # never materialised. The script does warn, but the line is easy to miss in the
  # start banner.
  #
  # With _CTX_HINT=65536 the 3 holds: 4 copies x (4.0 GiB + 152 MiB GDN) =
  # 16.6 GiB + 16.70 GiB base = 33.3 of 38.0 GiB usable, 4.7 GiB spare. That
  # gives each of the three instances its own warm snapshot slot.
  #
  # THE 3 DEPENDS ON THE CLIENTS. Raising one instance to 98304 requires pulling
  # APC_ENTRIES back with it -- otherwise the guard silently caps to 1 and ALL
  # three lose their warm slot. The mem lines in the log stay authoritative, not
  # this calculation.
  #
  # Side finding from the same measurement: KV_BITS=8 does NOT attack the
  # consumer. apc_adapters.py:515 calls dequantize_for_apc() on snapshot store --
  # the live cache shrinks to 32 KiB/token, the snapshots stay f16 at 64. It cost
  # 22.9 -> 18.7 tok/s decode (mean of 6 and 8 requests respectively).
  # That is why _KV_BITS stays empty here.
  roomy)
    _APC_ENTRIES=3
    if [[ "${_WIRED_MB:-0}" -ge 40960 ]]; then _PREFILL=2048; else _PREFILL=512; fi
    _KV_BITS="";  _VISION=20; _APC_MAXGB=80; _APC_MINFREE=2.0; _CTX_HINT=65536 ;;
  *)
    echo "ERROR: unknown PROFILE='$PROFILE' (valid: auto | lean | balanced | roomy)" >&2; exit 1 ;;
esac

# ── The wired limit is missing: say so out loud ───────────────────────────────
# The fallback above is SILENT. sysctl is not persistent, and without the
# LaunchDaemon every reboot restores the macOS default (48 GB -> 36864). _PREFILL
# then falls to 512, and because the NAX path from PR #3842 requires qL >= 1024
# it no longer engages -- without anything standing out anywhere. That is exactly
# how the gain from patch 0013 gets lost between two restarts.
_WIRED_WANT=$(( _RAM_MB > 32768 ? _RAM_MB - 8192 : _RAM_MB - 6144 ))
if [[ "$PROFILE" == "roomy" && "${_WIRED_MB:-0}" -lt 40960 ]]; then
  echo "  ⚠️  iogpu.wired_limit_mb = ${_WIRED_MB:-<not set>} (< 40960)." >&2
  echo "      Profile roomy therefore runs in reduced mode: PREFILL_STEP ${_PREFILL}" >&2
  echo "      instead of 2048, and the NAX path (qL >= 1024) does not engage." >&2
  echo "      Now:        sudo ./set-iogpu-wired-limit.sh        # -> ${_WIRED_WANT}" >&2
  echo "      Persistent: sudo ./install-wired-limit-daemon.sh (sysctl alone" >&2
  echo "                  does not survive a reboot)." >&2
fi

# ── Speculative decoding (MTP) ────────────────────────────────────────────────
# ON BY DEFAULT. Measured through on the 48 GB machine on 2026-08-17:
#   throughput: decode 16.9-18.3 -> 26.9-41.5 t/s (+58..+132%)
#   acceptance: 42% prose, 90% JSON, 93% tool call
#   quality   : 7/7 answers BIT-IDENTICAL to the run without a drafter (temp 0)
#   APC       : survives (turn 2 cached 3178 of 3217) -- the cached_tokens=0
#               problem from mlx-vlm 0.6.12 no longer exists in 0.6.13.
# Especially attractive on 32 GB: the drafter costs only 0.23 GiB and the gain is
# pure bandwidth -- and bandwidth is scarce on the M5 base.
# Rollback: ENABLE_SPEC_DECODE=0 ./start-mlx_qwen3.8.sh
ENABLE_SPEC_DECODE="${ENABLE_SPEC_DECODE:-1}"
# DRAFT_KIND=mtp|dflash
#   mtp     MTP head, 0.23 GiB. Former default, still maintained:
#           +58..132% decode, acceptance 90% JSON / 93% tool call,
#           output 7/7 bit-identical.
#   dflash  DEFAULT since 2026-08-20. DFlash 2 (z-lab), 1.01 GiB in 4bit.
#           Measured against MTP on identical prompts (decode from predicted_ms):
#             JSON  37.8 -> 44.2 t/s   code 35.8 -> 42.6   long ctx 36.0 -> 40.8
#             prose 29.8 -> 29.9 t/s   (MTP is ahead on acceptance there)
#           So the gain sits in STRUCTURED output -- tool calls, JSON, code --
#           and therefore exactly in the agent workload. The acceptance RATE is
#           the same for both (median 80% vs 81%); DFlash 2 drafts more tokens
#           per round (block_size 4 instead of 3) and wins through that.
#           A block-diffusion drafter with a path selector. Needs local patch
#           0040 (the upstream PR #1959; it replaced the earlier own patch 0020)
#           -- mlx-vlm itself implements only DFlash v1.
#           CAREFUL WITH BLOCK SIZE: the checkpoint is designed for block_size 8,
#           but z-lab recommends <= 5 for quantized MLX models, and our own
#           kernel measurement shows a dispatch cliff at M=5
#           (M=1 5.7 ms, M=4 6.3 ms, M=5 7.4 ms). Hence a default of 4 here.
DRAFT_KIND="${DRAFT_KIND:-dflash}"
case "$DRAFT_KIND" in
  mtp)    _DRAFT_DEFAULT="$MODELS_ROOT/Qwen3.8-27B-MTP-4bit";      _BLOCK_DEFAULT="" ;;
  dflash) _DRAFT_DEFAULT="$MODELS_ROOT/Qwen3.8-27B-DFlash2-4bit";  _BLOCK_DEFAULT="4" ;;
  *) echo "ERROR: unknown DRAFT_KIND='$DRAFT_KIND' (valid: mtp | dflash)" >&2; exit 1 ;;
esac
DRAFT_MODEL="${DRAFT_MODEL:-$_DRAFT_DEFAULT}"
DRAFT_BLOCK_SIZE="${DRAFT_BLOCK_SIZE-$_BLOCK_DEFAULT}"

# ── Automatic prefix caching ──────────────────────────────────────────────────
# Reuses the KV cache when the new prompt contains the old one as a prefix
# (= every follow-up turn). Correct upstream since mlx-vlm 0.6.13.
ENABLE_APC="${ENABLE_APC:-1}"
# PER PROFILE (1 / 2 / 3), where Qwen3.6 used to sit at 8: a snapshot is as large
# as the prompt, and this model needs 64 KiB/token. Every additional entry is
# directly less context -- see the budget calculation in the banner. Even on
# 48 GB, going from 2 to 4 cost roughly a third of the budget (142k -> 85k
# tokens), and the loss on a cache miss is small thanks to the SSD tier
# (~350 ms restore).
# Together with APC_SINGLE=1 (below), N means N conversations kept warm.
APC_ENTRIES="${APC_ENTRIES:-$_APC_ENTRIES}"
# SSD tier: survives server restarts, measured factor 256 on a cold 36k prefill.
# Even more valuable on this machine because the prefill is slower. Price:
# ~2.3 GB of write load per 36k conversation -> cap below.
APC_DISK="${APC_DISK:-$STATE_DIR/apc}"
APC_DISK_MAX_GB="${APC_DISK_MAX_GB:-$_APC_MAXGB}"
# The cap is only an upper bound -- it also has to fit on the disk.
# Measured on 2026-08-20: the tier sat at 58 GB, exactly at the 60 GB cap, and
# evicted on practically every store ("APC disk: evicted 6 shard(s); now 56341.8
# MB / 64424.5 MB cap"). That makes the fallback layer for evicted RAM snapshots
# leaky, and precisely then a miss costs the full 50-70 s.
# roomy therefore goes to 80 GB -- but only as far as the volume carries it.
_APC_RESERVE_GB=25

# ── Dead namespace: GC ────────────────────────────────────────────────────────
# apc.py:221 (apc_disk_namespace) fingerprints the directory name from the model
# path, the adapter AND the KV quantisation. Every change to KV_BITS therefore
# creates a NEW namespace -- and mlx-vlm never reclaims the old one: the store
# sets `self.dir = root/<namespace>` (apc.py:892) and _rebuild_index() only globs
# inside it (apc.py:1113). Eviction cannot see foreign namespaces.
# FOUND on 2026-08-24: 10 GB in ...-0c1f0816 (dated 21 Aug, the KV_BITS=8 hash
# from the sampler measurement) next to 53 GB in the active ...-e063a74c, on a
# volume that was 91% full. So the tier thrashes against the 80 GB cap of the
# ACTIVE namespace while 10 GB sits dead beside it.
# The namespace with the newest mtime counts as active; it is never collected.
# A running server writes constantly, so its directory is always the newest --
# which makes the rule robust against a second instance as well.
APC_NS_KEEP_DAYS="${APC_NS_KEEP_DAYS:-3}"
_ns_active=""
if [[ -d "$APC_DISK" ]]; then
  _ns_all=( "$APC_DISK"/*(N/) )
  if (( ${#_ns_all} > 0 )); then
    _ns_active_mt=-1
    for _d in "${_ns_all[@]}"; do
      _mt=$(stat -f %m "$_d" 2>/dev/null || echo 0)
      if (( _mt > _ns_active_mt )); then _ns_active_mt=$_mt; _ns_active="$_d"; fi
    done
  fi
  if (( ${#_ns_all} > 1 )); then
    _now=$(date +%s); _freed_mb=0
    for _d in "${_ns_all[@]}"; do
      [[ "$_d" == "$_ns_active" ]] && continue
      # Defensive: only delete below $APC_DISK.
      [[ "$_d" == "$APC_DISK"/?* ]] || continue
      _mt=$(stat -f %m "$_d" 2>/dev/null || echo 0)
      # Whole days, truncated -- when in doubt keep rather than delete.
      _age_d=$(( (_now - _mt) / 86400 ))
      # In MB, not GB: a 400 MB namespace should not show up as "0 GB".
      _sz_mb=$(( $(du -sk "$_d" 2>/dev/null | cut -f1) / 1024 ))
      if (( _age_d >= APC_NS_KEEP_DAYS )); then
        echo "  APC GC: removed namespace ${_d:t} (${_sz_mb} MB, ${_age_d} days inactive)" >&2
        rm -rf -- "$_d" && _freed_mb=$(( _freed_mb + _sz_mb ))
      else
        echo "  Note: APC namespace ${_d:t} (${_sz_mb} MB) is inactive but only" \
             "${_age_d} days old -- kept (APC_NS_KEEP_DAYS=${APC_NS_KEEP_DAYS})." >&2
        echo "        Clean now: APC_NS_KEEP_DAYS=0 $ZSH_ARGZERO" >&2
      fi
    done
    (( _freed_mb > 0 )) && echo "  APC GC: ${_freed_mb} MB freed." >&2
  fi
fi

# The cap is an upper bound PER NAMESPACE -- not per volume. Only the ACTIVE
# namespace may therefore count as reusable: bytes of foreign namespaces are
# already counted as occupied by `df`, and eviction cannot reclaim them. Until
# 2026-08-24 `du -sk "$APC_DISK"` summed ALL namespaces here and overestimated
# the headroom by exactly their size -- on this machine by 10 GB (63 instead of
# 53), so a 125 GB cap instead of the correct 115.
if [[ -n "$_ns_active" && -d "$_ns_active" ]]; then
  _apc_used_gb=$(( $(du -sk "$_ns_active" 2>/dev/null | cut -f1) / 1048576 ))
else
  _apc_used_gb=0
fi
# df AFTER the GC so that freed bytes are counted.
_apc_free_gb=$(( $(df -k "${APC_DISK:h}" 2>/dev/null | tail -1 | awk '{print $4}') / 1048576 ))
_apc_cap_max=$(( _apc_used_gb + _apc_free_gb - _APC_RESERVE_GB ))
if [[ "$_apc_cap_max" -lt 10 ]]; then _apc_cap_max=10; fi
if [[ "$APC_DISK_MAX_GB" -gt "$_apc_cap_max" ]]; then
  echo "  Note: APC_DISK_MAX_GB ${APC_DISK_MAX_GB} -> capped to ${_apc_cap_max} GB" \
       "(only ${_apc_free_gb} GB free, ${_APC_RESERVE_GB} GB reserve)" >&2
  APC_DISK_MAX_GB=$_apc_cap_max
fi
# The disk tier only restores while at least this much RAM is still free. The
# upstream default of 2.0 is too tight for 32 GB -- a restore straight into
# memory pressure is exactly the road to "[METAL] Insufficient Memory".
APC_MIN_FREE_RAM_GB="${APC_MIN_FREE_RAM_GB:-$_APC_MINFREE}"
# Suppresses the redundant full snapshot (the checkpoint at len-16 is always the
# one that hits). Halves APC memory. Needs local patch 0010
# (patches/apply-patches.sh) -- without it the variable has no effect and the
# script warns.
APC_SINGLE="${APC_SINGLE:-1}"

# ── Prefill / slots ───────────────────────────────────────────────────────────
# 1024 INSTEAD OF 2048 (the 48 GB script): the prefill chunk determines the
# transient activation peak. On the 48 GB machine it was measured that prefill is
# compute-bound rather than chunk-bound (2048 vs 4096: 94.6 s vs 94.2 s on the
# same prompt) -- so halving it costs almost nothing and buys headroom. With
# plenty of budget: PREFILL_STEP=2048 ./start-...
# Remember whether the value came from the caller: only the profile default may
# be raised below when the fused path decouples the chunk size.
_PREFILL_FROM_ENV=0
[[ -n "${PREFILL_STEP:-}" ]] && _PREFILL_FROM_ENV=1
PREFILL_STEP="${PREFILL_STEP:-$_PREFILL}"
# 1 slot. On a dense model, batch 2 would scale almost linearly in aggregate
# (shared weight read), but every additional sequence costs a complete KV set +
# GDN state -- not affordable on 32 GB.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
# 20 -> 4: vision features are cached image embeddings, pure memory load here.
VISION_CACHE="${VISION_CACHE:-$_VISION}"
LOG_PROGRESS="${LOG_PROGRESS:-10}"
# Seconds between two memory measurements in the log; 0 disables the sampler.
# Rationale and implementation are further down, at the exec.
_MEM_PROBE_INTERVAL="${MEM_PROBE_INTERVAL:-5}"

# ── KV quantisation ───────────────────────────────────────────────────────────
# OFF BY DEFAULT, but the most important lever on this machine: KV_BITS=8 halves
# 64 -> 32 KiB/token and thereby doubles the possible context.
#     KV_BITS=8 QUANT_KV_START=8192 ./start-mlx_qwen3.8.sh
# QUANT_KV_START leaves the first 8k tokens unquantized -- short turns stay fully
# fast, only long ones pay the dequantisation cost.
# WHY NOT ON BY DEFAULT: for llama.cpp/Qwen3.6 it was measured on 2026-08-11 that
# KV quantisation COSTS up to 8x prefill and 1.9x decode (dequant cost grows with
# KV length). For the MLX path this has NOT been re-measured. On a machine with
# ~9 t/s decode such a factor would be fatal -- so measure first, then enable:
#     A/B with the same prompt, compare decode t/s from the log.
# If the context stays <= 40k anyway: leave it off, nothing is gained.
# ${VAR-...} without a colon: the profile applies only when KV_BITS is UNSET.
# Empty is a valid value here (= f16), which is why
#   PROFILE=lean KV_BITS= ./start-mlx_qwen3.8.sh
# must be able to switch quantisation back off.
KV_BITS="${KV_BITS-$_KV_BITS}"
KV_SCHEME="${KV_SCHEME:-}"
QUANT_KV_START="${QUANT_KV_START:-8192}"

# ── DO NOT USE: --max-kv-size ─────────────────────────────────────────────────
# mlx-vlm can hard-cap the KV cache (rotating window). That is tempting on 32 GB
# and still wrong: the equivalent idea (a sliding window only on the 16 full-attn
# layers, GDN state complete) was DISPROVED on 2026-08-17 with needle_hybrid.py --
# the needle outside the window was lost, and in the 40% case the model even
# hallucinated a wrong number (8347 instead of 8342). A small context plus
# compaction in the agent is better than a silent loss of quality.

# ── Checks ────────────────────────────────────────────────────────────────────
[[ -x "$VENV_PY" ]] || {
  echo "ERROR: venv Python not found: $VENV_PY"
  echo "       Set it up first:  ./install-prereqs.sh"
  exit 1
}
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: model not found: $MODEL_DIR"
  echo "       ./download-mlx-model.sh mlx-community/Qwen3.8-27B-4bit $MODEL_DIR"
  exit 1
fi

# Check completeness: are all shards referenced in the index file present?
# (Otherwise an aborted 15 GB download only shows up after the model load.)
if [[ -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  MISSING=$(python3 -c "
import json,os
d=json.load(open('$MODEL_DIR/model.safetensors.index.json'))
want=sorted(set(d['weight_map'].values()))
print(' '.join(f for f in want if not os.path.exists(os.path.join('$MODEL_DIR',f))))")
  if [[ -n "$MISSING" ]]; then
    echo "ERROR: model incomplete -- missing shards: $MISSING"
    echo "       ./download-mlx-model.sh mlx-community/Qwen3.8-27B-4bit $MODEL_DIR"
    exit 1
  fi
fi

if lsof -iTCP:$PORT -sTCP:LISTEN -n &>/dev/null; then
  echo "WARNING: port $PORT is already in use."
  echo "  -> stop mlx-vlm:      pkill -f mlx_vlm.server"
  echo "  -> stop llama-server: pkill -f llama-server"
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

# ── Patch checks ──────────────────────────────────────────────────────────────
SITE_PACKAGES=$("$VENV_PY" -c "import mlx_vlm,os;print(os.path.dirname(os.path.dirname(mlx_vlm.__file__)))")
MLX_VLM_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx-vlm'))" 2>/dev/null || echo "?")
MLX_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx'))" 2>/dev/null || echo "?")

# APC capability: semantic_extra_hash() exists from 0.6.13. If it is missing
# (downgrade), the prefix cache only hits on byte-identical prompts -- while
# still costing memory, and memory is the scarce resource here.
if [[ "$ENABLE_APC" == "1" ]] && ! grep -q "def semantic_extra_hash" "$SITE_PACKAGES/mlx_vlm/apc.py" 2>/dev/null; then
  echo "⚠️  WARNING: mlx-vlm $MLX_VLM_VER does not know semantic_extra_hash() (< 0.6.13?)." >&2
  echo "    Prefix caching then hits ONLY on byte-identical prompts." >&2
  echo "    Fix:  uv pip install -U mlx-vlm" >&2
fi
# Patch 0010 (single snapshot). Without it APC_SINGLE has no effect and the
# memory need per conversation is twice what is computed below.
APC_SINGLE_OK=0
grep -q "QWEN38_APC_SINGLE_SNAPSHOT" "$SITE_PACKAGES/mlx_vlm/generate/ar.py" 2>/dev/null && APC_SINGLE_OK=1
if [[ "$APC_SINGLE" != "0" && "$APC_SINGLE_OK" == "0" ]]; then
  echo "⚠️  WARNING: patch 0010 missing -- APC_SINGLE has no effect, every request" >&2
  echo "    stores TWO snapshots (double APC memory, half the context)." >&2
  echo "    Fix:  ./patches/apply-patches.sh" >&2
fi

# Patch 0013 (fused full attention). head_dim is 256, and without the fused path
# EVERY one of the 16 full_attention layers materialises a score transient of
# n_heads x qL x kL x 2 B. At 24 heads, chunk 512 and 23k context that is 533 MiB
# per layer -- this is the real prefill ceiling, and it does NOT appear in the KV
# budget calculation below. The patch can only take effect when mlx knows
# force_fused at all (>= 0.32.2); below that it is deliberately inert.
# Hence a runtime probe rather than a plain grep: "the patch is in the venv" and
# "the path is active" are two different statements here.
FUSED_OK=0
_FUSED_PROBE=no
_PATCH0013=0
if grep -q "_FORCE_FUSED_HEAD_DIMS" "$SITE_PACKAGES/mlx_vlm/models/base.py" 2>/dev/null; then
  _PATCH0013=1
  _FUSED_PROBE=$("$VENV_PY" -c '
import mlx.core as mx
q = mx.zeros((1, 1, 1, 128), dtype=mx.bfloat16)
try:
    mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, mask=None, force_fused=False)
    print("ok")
except TypeError:
    print("no")
' 2>/dev/null || echo no)
fi

if [[ "${QWEN38_FORCE_FUSED_SDPA:-1}" == "0" ]]; then
  FUSED_STATUS="unfused (disabled via QWEN38_FORCE_FUSED_SDPA=0)"
elif [[ "$_PATCH0013" == "0" ]]; then
  FUSED_STATUS="unfused (patch 0013 missing -- ./patches/apply-patches.sh)"
elif [[ "$_FUSED_PROBE" == "ok" ]]; then
  FUSED_STATUS="fused (patch 0013 active)"
  FUSED_OK=1
else
  FUSED_STATUS="unfused (patch 0013 inert: mlx $MLX_VER has no force_fused, needs >= 0.32.2)"
fi

# ── Fused quantized linears (patch 0015) ─────────────────────────────────────
# _fused_quantized_linears() holds a SECOND, concatenated copy of the quantized
# weights on the module permanently. That is the fixed floor which killed the
# sessions on 2026-08-21/22: it appears on the FIRST generation, is independent
# of context length (16 tokens trigger it just as much as 44,452) and is never
# released.
# MEASURED on a 40 GiB working set, idle after 5 requests:
#                       with fusion   without fusion   decode (mean of 5 each)
#   with spec decode      26.00 GiB       17.00 GiB     26.1 vs 25.7 tok/s
#   without spec decode   17.08 GiB       14.96 GiB     18.4 vs 18.2 tok/s
# 9 GiB against 1.5%, and the spread of both decode series overlaps completely.
# On this machine the memory is worth more.
# Back to upstream behaviour:  QWEN38_FUSED_LINEARS=1 ./start-mlx_qwen3.8.sh
export QWEN38_FUSED_LINEARS="${QWEN38_FUSED_LINEARS:-0}"
QLIN_STATUS="unfused (patch 0015, saves ~9 GiB)"
if ! grep -q "QWEN38_FUSED_LINEARS" \
     "$SITE_PACKAGES/mlx_vlm/models/qwen3_5/language.py" 2>/dev/null; then
  QLIN_STATUS="fused -- patch 0015 MISSING (~9 GiB floor)"
  if [[ "$QWEN38_FUSED_LINEARS" == "0" ]]; then
    echo "⚠️  WARNING: patch 0015 missing -- QWEN38_FUSED_LINEARS=0 has no effect." >&2
    echo "    The server occupies ~9 GiB more than necessary. Fix: ./patches/apply-patches.sh" >&2
  fi
elif [[ "$QWEN38_FUSED_LINEARS" != "0" ]]; then
  QLIN_STATUS="fused (upstream default, ~9 GiB floor)"
fi

# With fused full attention the score transient disappears -- exactly the reason
# roomy drops to chunk 512 without a wired limit. On top of that the NAX path
# (PR #3842) explicitly requires qL >= 1024:
#   query_sequence_length >= 1024 && query_head_dim == 256 && do_causal
# So at 512 it does not engage at all. Only raise the profile default; an
# explicit value from the caller stays untouched.
_PREFILL_NOTE=""
if [[ "$FUSED_OK" == "1" && "$_PREFILL_FROM_ENV" == "0" && "$PREFILL_STEP" -lt 1024 ]]; then
  _PREFILL_NOTE="  (512 -> 1024: fused active, NAX needs qL >= 1024)"
  PREFILL_STEP=1024
fi

# KV_BITS=8 was briefly the roomy default on 2026-08-21 and is out again.
# The reasoning was: with fused attention the score transient disappears, so the
# 64 KiB/token dominate, so halve them. The first part is right, the conclusion
# is not -- apc_adapters.py:515 calls dequantize_for_apc() on snapshot store, so
# the APC snapshots remain f16. Only the live cache is quantized, and that is not
# the consumer.
# MEASURED: decode 22.9 -> 18.7 tok/s (mean of 6 and 8 requests respectively),
# while active climbed to 37.78 GiB unchanged and the same OOM arrived.
# For anyone who wants it anyway:  KV_BITS=8 QUANT_KV_START=8192 ./start-...

# ── Check the drafter ─────────────────────────────────────────────────────────
# DFlash 2 ships upstream since mlx-vlm 0.6.16 (PR #2014) at
# speculative/drafters/dflash2/. Local patch 0040 carried the earlier upstream
# PR #1959, which was closed unmerged in favour of #2014 -- it is gone.
#
# CAREFUL, THE OLD PATH STILL EXISTS: speculative/drafters/qwen3_dflash/dflash.py
# is still shipped and is the v1 drafter. Probing that file would pass on 0.6.16
# while saying nothing about DFlash 2 -- which is why the probe below targets the
# v2 module.
if [[ "$DRAFT_KIND" == "dflash" && "$ENABLE_SPEC_DECODE" != "0" ]]; then
  if [[ ! -f "$SITE_PACKAGES/mlx_vlm/speculative/drafters/dflash2/dflash2.py" ]]; then
    echo "⚠️  WARNING: DFlash 2 not present (needs mlx-vlm >= 0.6.16) -- falling back to MTP." >&2
    echo "    Fix:  uv pip install -U 'mlx-vlm>=0.6.16'" >&2
    DRAFT_KIND=mtp
    DRAFT_MODEL="$MODELS_ROOT/Qwen3.8-27B-MTP-4bit"
    DRAFT_BLOCK_SIZE=""
  elif ! grep -q "_speculative_batch_path_enabled" \
         "$SITE_PACKAGES/mlx_vlm/server/generation.py" 2>/dev/null; then
    echo "⚠️  WARNING: patch 0021 missing -- under dflash NO prefix cache applies" >&2
    echo "    (every turn pays the full prefill). Fix:  ./patches/apply-patches.sh" >&2
  fi
fi

SPEC_STATUS="OFF"
if [[ "$ENABLE_SPEC_DECODE" != "0" ]]; then
  if [[ -f "$DRAFT_MODEL/config.json" ]]; then
    SPEC_STATUS="ON"
  else
    echo "  WARNING: drafter not found ($DRAFT_MODEL) -- starting WITHOUT spec-dec."
    if [[ "$DRAFT_KIND" == "dflash" ]]; then
      echo "           ./download-mlx-model.sh z-lab/Qwen3.8-27B-DFlash2 ${MODELS_ROOT}/Qwen3.8-27B-DFlash2-bf16"
      echo "           ./convert-dflash2-drafter.py ${MODELS_ROOT}/Qwen3.8-27B-DFlash2-bf16 $DRAFT_MODEL"
    else
      echo "           ./download-mlx-model.sh mlx-community/Qwen3.8-27B-MTP-4bit $DRAFT_MODEL"
    fi
    ENABLE_SPEC_DECODE=0
  fi
fi

# ── Set the alias symlink ─────────────────────────────────────────────────────
ALIAS_LINK="$MODELS_ROOT/$MODEL_ALIAS"
if [[ -e "$ALIAS_LINK" && ! -L "$ALIAS_LINK" ]]; then
  echo "ERROR: $ALIAS_LINK exists and is NOT a symlink -- please check/remove it." >&2
  exit 1
fi
ln -sfn "$MODEL_DIR" "$ALIAS_LINK"

# ── Compute the memory budget ─────────────────────────────────────────────────
# Computes with the REAL values of this machine rather than the assumptions in
# the script header: Metal working set (follows iogpu.wired_limit_mb), actual
# file sizes, the chosen KV bits, the chosen APC entries.
WEIGHTS_KB=$(du -skL "$MODEL_DIR" | cut -f1)
DRAFT_KB=0
[[ "$ENABLE_SPEC_DECODE" != "0" ]] && DRAFT_KB=$(du -skL "$DRAFT_MODEL" | cut -f1)
# mlx-vlm has no -c flag: the native context is in config.json.
MODEL_CTX=$(python3 -c "
import json
d=json.load(open('$MODEL_DIR/config.json'))
tc=d.get('text_config',d)
print(tc.get('max_position_embeddings') or d.get('max_position_embeddings') or '')" 2>/dev/null || true)
BUDGET=$(
  WEIGHTS_KB="$WEIGHTS_KB" DRAFT_KB="$DRAFT_KB" KV_BITS="$KV_BITS" MODEL_CTX="$MODEL_CTX" \
  APC_ENTRIES="$APC_ENTRIES" ENABLE_APC="$ENABLE_APC" APC_SINGLE="$APC_SINGLE" \
  APC_SINGLE_OK="$APC_SINGLE_OK" CTX_HINT="$_CTX_HINT" PREFILL_STEP="$PREFILL_STEP" \
  FUSED_OK="$FUSED_OK" \
  "$VENV_PY" - <<'PY'
import os
import mlx.core as mx

GiB = 1 << 30
info = mx.device_info()
ws = info["max_recommended_working_set_size"]
ram = info["memory_size"]

weights = (int(os.environ["WEIGHTS_KB"]) + int(os.environ["DRAFT_KB"])) * 1024
# Activations, Metal heap fragmentation, tokenizer, Python. Empirical value from
# the 48 GB machine (RSS 15.5 GiB at idle with 15.2 GiB of weights, the peak
# grows with the prompt) -- deliberately generous, because underestimating here
# means OOM.
reserve = int(1.5 * GiB)

kv_bits = os.environ.get("KV_BITS") or ""
per_tok = 16 * 4 * (256 + 256) * 2                      # 65536 B, f16
if kv_bits:
    per_tok = int(per_tok * float(kv_bits) / 16.0)
# GDN/Mamba state: 48 linear layers x 48 v-heads x 128 (k) x 128 (v) x 4 B (float32)
recurrent = 48 * 48 * 128 * 128 * 4

apc_on = os.environ["ENABLE_APC"] == "1"
entries_req = int(os.environ["APC_ENTRIES"]) if apc_on else 0
model_ctx = int(os.environ.get("MODEL_CTX") or 0)
ctx_hint = int(os.environ.get("CTX_HINT") or 0)

# APC_EXACT_CACHE_ENTRIES caps the number of snapshots, not the bytes -- so the
# memory need is entries * prompt length, independent of patch 0010. The patch
# only changes HOW MANY CONVERSATIONS fit into those entries (with it: one per
# entry, without: one per two entries).
# ── Prefill transient ─────────────────────────────────────────────────────────
# The item that was missing until 2026-08-21, and the reason for "[METAL]
# Insufficient Memory" despite an apparently ample budget. Applies only WITHOUT
# the source build -- with patch 0013 active the item is zero (FUSED_OK branch
# below).
#
# head_dim is 256. mlx's default dispatch permits fused full attention only for
# 64/80/128 (up to 0.32.1), so the 16 full-attn layers run on the unfused graph
# and each materialise a score tensor of n_heads x qL x kL x 2 B. At
# PREFILL_STEP 2048 and 38k context that is 2.8 GiB -- PER LAYER. Delayed
# evaluation keeps several of them alive at once.
#
# INFLIGHT = 16, the number of full-attention layers: under delayed evaluation
# each of them may hold its score tensor at the same time in the worst case.
# CROSS-CHECKED on a 37.4 GiB working set (cold prefill, tier cleared first):
#   chunk  512  computed ~40k   measured through to ~38k, OOM above that
#   chunk 2048  computed ~12k   measured through to ~30k
# So at large chunks the calculation is too cautious -- which is the right
# direction to err. Underestimating here means a crash mid-operation, and it does
# not arrive as a clean error but as
# "[METAL] Command buffer execution failed: Insufficient Memory".
# The item disappears as soon as mlx 0.32.2 + patch 0013 provide the fused path --
# which is the case since the source build of 2026-08-21, hence the FUSED_OK
# branch.
# MEASURED with mlx 0.32.2.dev+a082cb91, production shape qL=512 / kL=22747,
# 24 heads / 4 KV heads / head_dim 256:
#   force_fused=False : peak 662 MiB   (difference 539 MiB = the score tensor)
#   force_fused=True  : peak 123 MiB
# With the fused path the item is therefore not "smaller" but gone.
INFLIGHT = 16
n_heads = 24
prefill_step = int(os.environ.get("PREFILL_STEP") or 2048)
if os.environ.get("FUSED_OK") == "1":
    transient_per_tok = 0
else:
    transient_per_tok = INFLIGHT * n_heads * prefill_step * 2

def budget(entries):
    cop = 1 + entries                                    # 1 live + snapshots
    av = ws - weights - reserve - cop * recurrent
    # The transient grows with the context length, not with the number of copies.
    tok = int(av / (cop * per_tok + transient_per_tok)) if av > 0 else 0
    if model_ctx:
        tok = min(tok, model_ctx)
    return cop, av, tok

# ── Prevent overbooking ───────────────────────────────────────────────────────
# A profile targets a particular working set (roomy for 40 GiB with wired_limit
# set). If the limit is not set, PROFILE=auto still picks roomy -- and then the
# snapshot count no longer matches the memory.
# The symptom is not a clean error but "[METAL] Command buffer execution failed:
# Insufficient Memory" mid-operation, often only after several requests.
# Hence: the recommended context length has to fit the budget with MARGIN to
# spare; otherwise the snapshots are reduced step by step. Measured on a 37.4 GiB
# machine: with 1.5 GiB of reserve every fourth request died, with 7.5 GiB it ran
# through.
#
# CAREFUL, THIS GUARD IS SILENT BY DESIGN and its input is _CTX_HINT, not the
# real client context. If _CTX_HINT is set too high, entries_used collapses to 1
# regardless of what APC_ENTRIES says -- see "WHY 3" in the profile block.
MARGIN = 0.20                                            # 20% headroom on the budget
entries_used = entries_req
copies, avail, tokens = budget(entries_used)
if apc_on and ctx_hint:
    while entries_used > 1 and ctx_hint > tokens * (1 - MARGIN):
        entries_used -= 1
        copies, avail, tokens = budget(entries_used)

print(f"{ws/GiB:.1f}|{ram/GiB:.0f}|{weights/GiB:.1f}|{avail/GiB:.1f}|{tokens}|{copies}"
      f"|{per_tok//1024}|{entries_used}|{info['device_name']}")
PY
)
WS_GIB="${BUDGET%%|*}"; REST="${BUDGET#*|}"
RAM_GIB="${REST%%|*}"; REST="${REST#*|}"
W_GIB="${REST%%|*}";   REST="${REST#*|}"
AVAIL_GIB="${REST%%|*}"; REST="${REST#*|}"
MAX_TOKENS_FIT="${REST%%|*}"; REST="${REST#*|}"
COPIES="${REST%%|*}";  REST="${REST#*|}"
KV_KIB="${REST%%|*}";  REST="${REST#*|}"
ENTRIES_USED="${REST%%|*}"; DEV_NAME="${REST#*|}"

# If the budget calculation capped the snapshot count, the capped value applies.
if [[ "$ENABLE_APC" == "1" && "$ENTRIES_USED" != "$APC_ENTRIES" ]]; then
  echo "⚠️  APC_ENTRIES $APC_ENTRIES -> capped to $ENTRIES_USED: profile '$PROFILE' targets more" >&2
  echo "    working set than this machine has (${WS_GIB} GiB). With $APC_ENTRIES snapshots," >&2
  echo "    context_length $_CTX_HINT would have no reserve -- that ends in" >&2
  echo "    '[METAL] Insufficient Memory' during operation, not in a clean error." >&2
  echo "    More working set: sudo ./set-iogpu-wired-limit.sh (see README)." >&2
  echo "    NOT 45056 on 48 GB while a browser/IDE is running -- that leaves" >&2
  echo "    macOS ~2 GiB and ends in a kernel panic instead of a Metal OOM." >&2
  echo "    NOTE: the input to this guard is _CTX_HINT, not the real client context." >&2
  echo "    Override: APC_ENTRIES=$APC_ENTRIES ./start-mlx_qwen3.8.sh" >&2
  APC_ENTRIES="$ENTRIES_USED"
fi

# Score transient of unfused full attention. Has to sit BEFORE the banner and
# belongs next to the KV budget there: it grows with qL x kL, appears nowhere in
# the KV calculation, and without patch 0013 it is the quantity that drives the
# prefill into the wall first. Heads/layers come from config.json so the number
# stays correct after a model change.
SCORE_GIB="-"
if [[ "$FUSED_OK" == "0" ]]; then
  SCORE_GIB=$("$VENV_PY" -c '
import json, sys
cfg = json.load(open(sys.argv[1] + "/config.json"))
tc = cfg.get("text_config", cfg)
heads = tc.get("num_attention_heads") or 0
types = tc.get("layer_types") or []
full = sum(1 for t in types if t == "full_attention") or tc.get("num_hidden_layers", 0)
qL, kL = int(sys.argv[2]), int(sys.argv[3])
print("%.1f" % (heads * qL * kL * 2 * full / 1024 ** 3) if heads and full else "?")
' "$MODEL_DIR" "$PREFILL_STEP" "$_CTX_HINT" 2>/dev/null || echo "?")
fi

{
echo "──────────────────────────────────────────────────────────────"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  mlx-vlm $MLX_VLM_VER / mlx $MLX_VER  |  Qwen3.8 27B (DENSE, MLX 4bit)  |  $DEV_NAME"
echo "  Profile  :  $PROFILE$_AUTO_NOTE  -  recommended context_length $_CTX_HINT"
echo "  Model    :  $MODEL_DIR"
echo "  API name :  $MODEL_ALIAS  (symlink; MUST match the request model name)"
echo "  Port     :  $BIND_HOST:$PORT"
echo "  SpecDec  :  $SPEC_STATUS  (drafter: $DRAFT_KIND, $(du -shL "$DRAFT_MODEL" 2>/dev/null | cut -f1)${DRAFT_BLOCK_SIZE:+, block_size $DRAFT_BLOCK_SIZE})"
if [[ "$ENABLE_APC" == "1" ]]; then
  echo "  APC      :  ON ($APC_ENTRIES snapshots, single=$([[ "$APC_SINGLE" != "0" && "$APC_SINGLE_OK" == "1" ]] && echo yes || echo NO)${APC_DISK:+, SSD: $APC_DISK})"
else
  echo "  APC      :  OFF"
fi
echo "  KV cache :  ${KV_BITS:-f16 unquantized} -> ${KV_KIB} KiB/token"
echo "  Mem probe:  $([[ "$_MEM_PROBE_INTERVAL" == "0" ]] && echo "OFF" || echo "every ${_MEM_PROBE_INTERVAL}s to the log (MEM_PROBE_INTERVAL=0 disables)")"
echo "  Prefill  :  chunk $PREFILL_STEP   slots: $MAX_NUM_SEQS   vision cache: $VISION_CACHE$_PREFILL_NOTE"
echo "  Full-attn:  $FUSED_STATUS"
echo "  Q-linears:  $QLIN_STATUS"
echo "  Log      :  $LOG_FILE"
echo "  ──────────── memory budget (computed, not measured) ─────────"
echo "  RAM              : ${RAM_GIB} GiB"
echo "  Metal working set: ${WS_GIB} GiB   (follows iogpu.wired_limit_mb)"
echo "  weights          : ${W_GIB} GiB   (+1.5 GiB reserve for activations)"
echo "  free for KV      : ${AVAIL_GIB} GiB  across ${COPIES} copies (1 live + APC)"
echo "  ->  CONTEXT BUDGET: ~${MAX_TOKENS_FIT} tokens  -- UPPER BOUND, not a promise"
echo "     The calculation assumes (1+APC_ENTRIES) KV copies and underestimates"
echo "     the real need. Measured 2026-08-21: a request over 13k tokens cost"
echo "     +11.4 GiB instead of the computed 1.6, and active does not fall back"
echo "     between requests. The mem lines in the log are authoritative."
if [[ "$FUSED_OK" == "0" ]]; then
echo "  score transient  : ${SCORE_GIB} GiB  at chunk $PREFILL_STEP @ ${_CTX_HINT} tokens"
echo "                     (unfused; comes ON TOP of the KV budget above)"
fi
echo "──────────────────────────────────────────────────────────────"

# Hard warning when the working set sits at the 32 GB default.
if (( $(printf '%.0f' "$WS_GIB") < 24 )) && (( $(printf '%.0f' "$RAM_GIB") >= 30 )); then
  echo "ℹ️  Working set is ${WS_GIB} GiB -- that is the macOS default (2/3 of RAM)."
  echo "    With  sudo sysctl -w iogpu.wired_limit_mb=26624  that becomes 26 GiB"
  echo "    and roughly doubles the ~${MAX_TOKENS_FIT} tokens. Persistent: see README."
fi
if [[ "$FUSED_OK" == "0" ]]; then
  echo "⚠️  Full attention is running unfused: $FUSED_STATUS"
  echo "    The prefill then tips into '[METAL] Insufficient Memory' long before"
  echo "    the CONTEXT BUDGET above is reached -- the score transient counts too."
  echo "    Do NOT raise PREFILL_STEP: the transient grows linearly with the chunk,"
  echo "    1024 doubles it against 512. First mlx >= 0.32.2, then the chunk."
fi
if [[ -z "$KV_BITS" && "$MAX_TOKENS_FIT" -lt 40000 ]]; then
  echo "ℹ️  Below a 40k token budget. Double it without sudo:"
  echo "      KV_BITS=8 QUANT_KV_START=8192 ./start-mlx_qwen3.8.sh   (measure A/B first)"
fi
} | tee -a "$LOG_FILE"

# ── Client reconciliation (optional, read-only) ───────────────────────────────
# mlx-vlm has NO -c flag; the context comes from config.json (262144). The
# effective ceiling is therefore the client's context_length alone. If that sits
# above the budget, the server eventually runs into
# "[METAL] Insufficient Memory".
#
# Anyone keeping a client configuration in YAML with a model: block
# (model.context_length, model.default) can have it cross-checked here:
#
#   CLIENT_CONFIG=~/path/to/config.yaml ./start-mlx_qwen3.8.sh
#
# Without the variable the block is inert. It changes nothing, it only warns.
CLIENT_CONFIG="${CLIENT_CONFIG:-}"
if [[ -n "$CLIENT_CONFIG" && -f "$CLIENT_CONFIG" ]]; then
  CONFIG_CTX=$(awk '
    /^model:/ { in_model=1; next }
    in_model && /^[a-zA-Z_]/ { in_model=0 }
    in_model && /context_length:/ { gsub(/[^0-9]/,"",$0); print; exit }
  ' "$CLIENT_CONFIG")
  if [[ -n "$CONFIG_CTX" && "$CONFIG_CTX" -gt "$MAX_TOKENS_FIT" ]]; then
    echo "⚠️  WARNING: ${CLIENT_CONFIG:t} model.context_length=$CONFIG_CTX > budget $MAX_TOKENS_FIT." >&2
    echo "    The client will build prompts that no longer fit in memory here." >&2
    echo "    Either lower context_length or raise KV_BITS=8 / wired_limit." >&2
  fi
  CONFIG_MODEL=$(awk '
    /^model:/ { in_model=1; next }
    in_model && /^[a-zA-Z_]/ { in_model=0 }
    in_model && /default:/ { sub(/^[^:]*:[[:space:]]*/,""); gsub(/["\x27]/,""); print; exit }
  ' "$CLIENT_CONFIG")
  if [[ -n "$CONFIG_MODEL" && "$CONFIG_MODEL" != "$MODEL_ALIAS" ]]; then
    echo "⚠️  WARNING: ${CLIENT_CONFIG:t} model.default='$CONFIG_MODEL' != alias '$MODEL_ALIAS'." >&2
    echo "    With mlx-vlm the request model name IS the load path -- a mismatch leads" >&2
    echo "    to a reload + HF download (401). Align them, or:" >&2
    echo "      MODEL_ALIAS='$CONFIG_MODEL' ./start-mlx_qwen3.8.sh" >&2
  fi
fi

# ── Thinking ──────────────────────────────────────────────────────────────────
# OFF on the server side (mlx-vlm always sends enable_thinking to the template,
# default false). It is controlled per request by the client.
#
# THE TEMPLATE knows only xhigh|medium|low and throws an exception on anything
# else -> HTTP 500. With nothing specified it defaults to 'xhigh'. Curiously only
# xhigh and low set an instruction; 'medium' sets none at all and therefore falls
# back to the model's default behaviour.
#
# MLX-VLM INTERCEPTS EARLIER. request_normalization.py:21:
#   _DISABLED_REASONING_EFFORTS = {"none", "off", "disabled", "false", "0"}
# These values never reach the template, they are translated into
# enable_thinking=false -- and the template turns that into a PRE-CLOSED block
# ('<think>\n\n</think>'), so the model does not think at all.
# The claim that used to stand here, that "none" also throws a 500, is DISPROVED:
# only unknown values such as "minimal" get through.
#
# MEASURED on 2026-08-23, same prompt, temperature 0:
#   reasoning_effort=none / off      76 tk   3.2 s   (does not think)
#   enable_thinking=false            76 tk   3.2 s   (identical)
#   reasoning_effort=low            234 tk   7.8 s
# For tool calls and short turns 'none' is the cheapest route; where the model
# really should think, 'low' remains. The client decides that per request, there
# is nothing to configure here.
#
# --thinking-budget would be the hard token limit INSIDE the thinking block, but
# it is blocked together with speculative decoding (generation.py:1257) -- using
# it would mean giving up the drafter.

args=(
  --host                  "$BIND_HOST"
  --port                  "$PORT"
  --model                 "$MODEL_ALIAS"
  --prefill-step-size     "$PREFILL_STEP"
  --max-num-seqs          "$MAX_NUM_SEQS"
  --vision-cache-size     "$VISION_CACHE"
  --log-progress-interval "$LOG_PROGRESS"
)

if [[ "$ENABLE_SPEC_DECODE" != "0" ]]; then
  args+=( --draft-model "$DRAFT_MODEL" --draft-kind "$DRAFT_KIND" )
  # Without this, non-MTP drafters run in their own generation loop that never
  # wires up the APC manager: cached_tokens=0 on EVERY turn. Needs patch 0021.
  # Measured: cached 0 -> 5748/5788, decode unchanged.
  # Overridable since 2026-08-21 so the path can be measured. It was suspected of
  # causing the fixed ~9.4 GiB memory floor -- MEASURED, IT IS NOT: with BATCH=0
  # the floor stays unchanged at +9.46 GiB. The floor hangs on
  # --draft-block-size >= 2 (1 -> +0.16 GiB, 2 -> +9.31).
  # BATCH=0 costs the prefix cache under dflash, so it is nothing for production
  # use -- the switch exists only for diagnostic runs.
  [[ "$DRAFT_KIND" != "mtp" ]] && export MLX_VLM_SPECULATIVE_BATCH="${MLX_VLM_SPECULATIVE_BATCH:-1}"
  [[ -n "$DRAFT_BLOCK_SIZE" ]] && args+=( --draft-block-size "$DRAFT_BLOCK_SIZE" )
fi

if [[ -n "$KV_BITS" ]]; then
  args+=( --kv-bits "$KV_BITS" )
  [[ -n "$KV_SCHEME"      ]] && args+=( --kv-quant-scheme "$KV_SCHEME" )
  [[ -n "$QUANT_KV_START" ]] && args+=( --quantized-kv-start "$QUANT_KV_START" )
fi

if [[ "$ENABLE_APC" == "1" ]]; then
  export APC_ENABLED=1
  export APC_EXACT_CACHE_ENTRIES="$APC_ENTRIES"
  if [[ -n "$APC_DISK" ]]; then
    mkdir -p "$APC_DISK"
    export APC_DISK_PATH="$APC_DISK"
    export APC_DISK_MAX_GB="$APC_DISK_MAX_GB"
    export APC_DISK_MIN_FREE_RAM_GB="$APC_MIN_FREE_RAM_GB"
  fi
fi
[[ "$APC_SINGLE" != "0" ]] && export QWEN38_APC_SINGLE_SNAPSHOT=1

# CWD = models root so the relative alias name resolves
# (get_model_path() does Path(name).exists() against the working directory).
cd "$MODELS_ROOT"

# ── Memory sampler ────────────────────────────────────────────────────────────
# Why at all: mlx puts every allocation into a residency set
# (mlx/backend/metal/allocator.cpp), i.e. makes it wired. The buffer cache is
# released at gc_limit_ = 0.95 x working set, but LIVE memory never is -- below
# the working set there is no brake on it. Once the sum exceeds the Metal
# ceiling, the next command buffer fails with "[METAL] Insufficient Memory", and
# at an arbitrary place: on 2026-08-21 once in attention (ar.py:1907), once in
# the APC clone (apc.py:321). The stack trace therefore shows the location, not
# the cause.
# From outside this is not measurable: RSS does not include the Metal buffers
# (2.3 GiB RSS with 16 GiB of weights). So from the inside, via a thread, without
# touching mlx_vlm -- a patch in site-packages would be gone after every mlx-vlm
# update.
# Disable: MEM_PROBE_INTERVAL=0 (the value is set above, near LOG_PROGRESS)
#
# NOTE FOR ANYONE REWORDING THE LINE BELOW: watchdog-mlx_qwen3.8.sh parses it.
# It matches "(NN%" up to the closing parenthesis and deliberately ignores the
# wording, so rephrasing is safe -- but the percentage must stay inside
# parentheses, otherwise the watchdog silently stops restarting anything.
_MEM_BOOT=$(cat <<'PY'
import os, sys, threading, time, logging, runpy

import mlx.core as mx

_GIB = 1024 ** 3


def _working_set():
    try:
        info = mx.device_info()
    except AttributeError:
        info = mx.metal.device_info()
    return float(info.get("max_recommended_working_set_size") or 0)


def _sample(interval, ws):
    log = logging.getLogger("memprobe")
    # Sleep first: mlx_vlm configures root logging at startup; before that the
    # line goes nowhere instead of into server.log.
    time.sleep(interval)
    while True:
        try:
            active = mx.get_active_memory()
            cache = mx.get_cache_memory()
            total = active + cache
            pct = (total / ws * 100.0) if ws else 0.0
            msg = (
                "mem active=%.2f cache=%.2f sum=%.2f GiB "
                "(%.0f%% of %.2f GiB working set) peak=%.2f GiB"
            )
            argv = (
                active / _GIB, cache / _GIB, total / _GIB,
                pct, ws / _GIB, mx.get_peak_memory() / _GIB,
            )
            # From 85% on, the next larger prefill is the likely trigger -- that
            # should stand out in the log rather than drown in INFO.
            log.warning(msg, *argv) if pct >= 85.0 else log.info(msg, *argv)
        except Exception:
            pass
        time.sleep(interval)


_iv = float(os.environ.get("MEM_PROBE_INTERVAL", "5") or 0)
if _iv > 0:
    threading.Thread(
        target=_sample, args=(_iv, _working_set()), daemon=True
    ).start()

sys.argv[0] = "mlx_vlm.server"
runpy.run_module("mlx_vlm.server", run_name="__main__")
PY
)

if [[ "$_MEM_PROBE_INTERVAL" != "0" ]]; then
  export MEM_PROBE_INTERVAL="$_MEM_PROBE_INTERVAL"
  exec "$VENV_PY" -c "$_MEM_BOOT" "${args[@]}" > >(tee -a "$LOG_FILE") 2>&1
else
  exec "$VENV_PY" -m mlx_vlm.server "${args[@]}" > >(tee -a "$LOG_FILE") 2>&1
fi
