#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Prerequisites for Qwen3.8-27B on MLX  -  Apple Silicon, 32 GB and up
#
# Sets up everything start-mlx_qwen3.8.sh needs, starting from a fresh macOS:
# Xcode CLT -> uv -> venv (Python 3.12) -> mlx-vlm + patches -> model + MTP
# drafter -> directories. Idempotent: re-runnable, completed steps are skipped,
# downloads resume.
#
# Usage:
#   ./install-prereqs.sh                 # everything, with pinned versions
#   ./install-prereqs.sh --skip-model    # software only, no 15 GB download
#   ./install-prereqs.sh --latest        # newest versions instead of the pinned ones
#   ./install-prereqs.sh --check         # verify only, change nothing
#
# Env overrides:  MLX_HOME (default ~/src/mlx), MLX_MODELS, PYTHON_VERSION
#
# WHAT THIS SCRIPT DOES NOT DO: sudo. The wired limit (iogpu.wired_limit_mb) is
# the most important tuning step but changes system state -- the necessary
# commands are only PRINTED at the end, see also README.md.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

BUNDLE_DIR="${0:A:h}"
MLX_HOME="${MLX_HOME:-$HOME/src/mlx}"
MLX_MODELS="${MLX_MODELS:-$MLX_HOME/models}"
VENV_DIR="$MLX_HOME/.venv"
VENV_PY="$VENV_DIR/bin/python"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

SKIP_MODEL=0
PINNED=1
CHECK_ONLY=0
for a in "$@"; do
  case "$a" in
    --skip-model) SKIP_MODEL=1 ;;
    --latest)     PINNED=0 ;;
    --check)      CHECK_ONLY=1 ;;
    -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "Unknown option: $a"; exit 1 ;;
  esac
done

# Pinned state, VERIFIED as working on an M5 Pro / macOS 26 (2026-09-25).
#
# 0.7.3 (2026-09-24, tagged at 573562d) over 0.7.2: for this setup it adds only
# #2328 (the #2310 closure-cycle fix, measured harmless here). apc*.py,
# speculative/, models/qwen3_5/, models/cache.py, server/generation.py and
# tools/ are unchanged; all eight patches apply without rejects. It raises
# mlx-audio to >=0.5.2 (uv moved 0.4.8 -> 0.5.6, `uv pip check` clean) and keeps
# mlx>=0.32.2. Re-measured after the move: docs/upstream-2026-09-25.md.
#
# mlx-vlm is back on a PLAIN VERSION PIN. Between 2026-09-17 and 2026-09-21 it
# was pinned to a git commit (main @ 548b09b) because the 0.7.1 tag carried
# #2182 (the APC memory planner) without its fix #2262, and on a HYBRID model
# that combination silently kills exact APC: Qwen3.8-27B carries 48 GDN layers
# whose recurrent state does not scale with tokens, so one short prompt sets a
# bytes/token ratio ~10x too high, every later prefill over-reserves, and the
# manager stops storing AND restoring for the rest of the process (#2259). This
# server's traffic is short agent turns, so the tag would have lost the prefix
# cache within minutes of every start, without an error in the log.
#
# 0.7.2 (2026-09-21, tagged at a74c7de) ends that. Checked before switching, by
# diffing the tag against the commit that had been running:
#     apc.py  apc_adapters.py  apc_coordinator.py
#     models/base.py  speculative/  server/generation.py
#   -> no differences at all.
# The entire APC and speculative-decoding surface -- everything the seven
# patches touch, and everything #2259 was about -- is byte-identical to
# 548b09b. This upgrade is therefore administrative: the same code, from a
# release instead of a commit. All seven patches apply to a74c7de without fuzz.
#
# What the tag adds on top of the commit: #2291 (Anthropic tool-call turns with
# no text no longer crash the chat template), #2260 (Chat/Responses parity),
# #2320 (400 instead of a TypeError when a chat model is sent to /v1/audio/*),
# #2286 (a named error instead of AttributeError for caches that cannot batch),
# plus new models. One behaviour change to know about: model discovery now runs
# by DEFAULT, so /v1/models also lists what is in the HF cache, and the
# --model-discovery flag is gone (--model-dir replaces it; we use neither).
# What it does NOT add: #2310, the KV-cache leak in
# GenerationBatch._eval_pending_state, is still present in 0.7.2.
#
# NEVER go back to 0.7.1 -- it is the one release this setup cannot run. The
# start script tests for the MECHANISM (_bytes_per_token in apc.py) rather than
# the version string, and warns.
#
# mlx 0.32.2 is on PyPI since 2026-08-25, including mlx-metal and
# macosx_26_0_arm64 wheels -- the source build documented in docs/build-mlx.md
# is no longer needed. Patch 0013 is still required (the default dispatch still
# does not route to force_fused) and applies unchanged; re-verified on
# 548b09b through the patched entry point with the real server mask, fused at
# qL 512/1024/2048.
#
# UPGRADE CAREFULLY: every install wipes the patches out of site-packages --
# run ./patches/apply-patches.sh afterwards.
# --latest gets you something newer; apply-patches.sh may then report "CONFLICT"
# (meaning: merged upstream -> delete the patch) and the measured values in the
# start script no longer hold unexamined.
# mlx-vlm is back inside PINS since 0.7.2. It declares `mlx>=0.32.2`, a lower
# bound, so resolving it together with the exact mlx pin below keeps 0.32.2 --
# the reason the git requirement needed its own --no-deps install (a git
# requirement let uv reconsider mlx, drop it to 0.32.1, and silently disable
# patch 0013) no longer applies.
PINS=(
  "mlx==0.32.2"
  "mlx-lm==0.31.3"
  "mlx-vlm==0.7.3"
  "transformers==5.15.1"
  "numpy==2.5.2"
  "huggingface-hub==1.27.0"
  "pillow==12.3.0"
)

MODEL_REPO="mlx-community/Qwen3.8-27B-4bit"
MODEL_DIR="$MLX_MODELS/Qwen3.8-27B-MLX-4bit"
DRAFT_REPO="mlx-community/Qwen3.8-27B-MTP-4bit"
DRAFT_DIR="$MLX_MODELS/Qwen3.8-27B-MTP-4bit"
MODEL_ALIAS="${MODEL_ALIAS:-Qwen3.8-27B-local}"
# Download size: model ~15.0 GiB (3 shards + tokenizer), drafter ~0.25 GiB.
NEEDED_GB=20

ok()   { echo "  ✓ $*" }
info() { echo "  · $*" }
warn() { echo "  ⚠️  $*" >&2 }
die()  { echo "  ✗ $*" >&2; exit 1 }

echo "──────────────────────────────────────────────────────────────"
echo "  Qwen3.8-27B / MLX  -  setup for Apple Silicon (32 / 48 GB)"
echo "  Target venv : $VENV_DIR"
echo "  Models      : $MLX_MODELS"
echo "──────────────────────────────────────────────────────────────"

# ── 1. Hardware/OS ────────────────────────────────────────────────────────────
echo
echo "[1/8] Hardware & macOS"
[[ "$(uname -s)" == "Darwin" ]] || die "Not macOS."
[[ "$(uname -m)" == "arm64" ]]  || die "Not Apple Silicon (uname -m = $(uname -m)). MLX needs arm64."
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "?")
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
ok "$CHIP · ${RAM_GB} GB RAM · macOS $(sw_vers -productVersion)"
# Wired-limit recommendation: RAM minus a reserve for macOS, computed in
# absolute terms (6 GiB up to 32 GB of RAM, 8 GiB above). Same rule as in
# set-iogpu-wired-limit.sh -- the rationale lives there.
#
# CAREFUL, A DANGEROUS VALUE USED TO LIVE HERE: until 2026-08-24 this branch
# suggested 45056 for RAM >= 44 GB. On a 48 GB machine that is only 4 GiB of
# reserve, and that exact value is what drove the test machine into a kernel
# panic on 2026-08-21 (watchdog timeout, see README). It also contradicted the
# plist, which set 26624 -- two different wrong values in one set of
# instructions. Both now come from the same script.
if (( RAM_GB <= 32 )); then
  WIRED_SUGGEST=$(( RAM_GB * 1024 - 6144 ))
else
  WIRED_SUGGEST=$(( RAM_GB * 1024 - 8192 ))
fi
# Below the macOS default (2/3 of RAM) the intervention would be a REGRESSION.
# Only affects very small machines (16 GB: 10240 < 10922), which per the warning
# below do not carry this model anyway. set-iogpu-wired-limit.sh refuses such
# values -- otherwise there would be a suggestion here that the script then
# rejects.
_WIRED_FLOOR=$(( RAM_GB * 1024 * 2 / 3 ))
(( WIRED_SUGGEST < _WIRED_FLOOR )) && WIRED_SUGGEST=$_WIRED_FLOOR
if   (( RAM_GB >= 44 )); then PROFILE_HINT="roomy"
elif (( RAM_GB >= 32 )); then PROFILE_HINT="balanced"
else                          PROFILE_HINT="lean"
fi
if (( RAM_GB < 32 )); then
  warn "Only ${RAM_GB} GB of RAM. The weights alone occupy 15.2 GiB -- below 32 GB"
  warn "no usable context is left. Choose a smaller model/quant."
elif (( RAM_GB == 32 )); then
  info "32 GB: tight but viable -- profile '$PROFILE_HINT'. See README.md."
else
  info "Profile '$PROFILE_HINT' fits this machine (PROFILE=auto picks it by itself)."
fi
FREE_GB=$(( $(df -k "$HOME" | awk 'NR==2{print $4}') / 1048576 ))
info "Free disk space: ${FREE_GB} GB (needed: ~${NEEDED_GB} GB for model+drafter,"
info "plus up to 40 GB for the APC SSD cache -- the cap lives in the start script)"
(( SKIP_MODEL == 1 || FREE_GB > NEEDED_GB )) || die "Not enough disk space."

# ── 2. Xcode Command Line Tools ───────────────────────────────────────────────
echo
echo "[2/8] Xcode Command Line Tools"
if xcode-select -p &>/dev/null; then
  ok "present ($(xcode-select -p))"
else
  warn "missing. The Metal toolchain is required. Install with:"
  echo "      xcode-select --install"
  (( CHECK_ONLY == 1 )) || die "Install the CLT first, then run this script again."
fi

# ── 3. uv ─────────────────────────────────────────────────────────────────────
echo
echo "[3/8] uv (package/venv manager)"
if command -v uv &>/dev/null; then
  ok "uv $(uv --version | awk '{print $2}')"
elif (( CHECK_ONLY == 1 )); then
  warn "uv missing"
else
  info "installing uv into ~/.local/bin (official installer from astral.sh)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv &>/dev/null || die "uv installation failed. Alternative: brew install uv"
  ok "uv $(uv --version | awk '{print $2}')"
  info "PATH entry for new shells:  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# ── 4. venv ───────────────────────────────────────────────────────────────────
echo
echo "[4/8] Virtualenv (Python $PYTHON_VERSION)"
if (( CHECK_ONLY == 1 )); then
  [[ -x "$VENV_PY" ]] && ok "$($VENV_PY -V)" || warn "venv missing: $VENV_DIR"
else
  mkdir -p "$MLX_HOME" "$MLX_MODELS"
  if [[ -x "$VENV_PY" ]]; then
    ok "present: $($VENV_PY -V)"
  else
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    ok "created: $($VENV_PY -V)"
  fi
fi

# ── 5. Pakete ─────────────────────────────────────────────────────────────────
echo
echo "[5/8] mlx-vlm & dependencies"
if (( CHECK_ONLY == 1 )); then
  [[ -x "$VENV_PY" ]] && "$VENV_PY" - <<'PY' || warn "venv missing"
import importlib.metadata as m
import json


def origin(pkg):
    """Where did this come from -- PyPI or a git commit?

    Since 0.7.2 the pin is a plain version again, so this normally prints
    nothing for mlx-vlm. It is kept because a git install is invisible in the
    version string: main @ 548b09b reported "0.7.1", the SAME string as the
    PyPI 0.7.1 tag that loses exact APC after the first short prompt (#2259).
    If a line below grows a "<- git ..." suffix, someone installed from source
    and the version number alone no longer says what is running.
    """
    try:
        dist = m.distribution(pkg)
        raw = dist.read_text("direct_url.json")
        if not raw:
            return ""
        data = json.loads(raw)
        vcs = data.get("vcs_info") or {}
        rev = vcs.get("commit_id") or vcs.get("requested_revision") or ""
        if rev:
            return f"  <- git {rev[:7]}"
        return f"  <- {data.get('url', 'local')}"
    except Exception:
        return ""


for p in ("mlx", "mlx-lm", "mlx-vlm", "transformers", "numpy", "huggingface-hub", "pillow"):
    try:
        print(f"  · {p:18s} {m.version(p)}{origin(p)}")
    except Exception:
        print(f"  ⚠️  {p:18s} MISSING")
PY
else
  if (( PINNED == 1 )); then
    info "pinned state (--latest for the newest versions)"
    VIRTUAL_ENV="$VENV_DIR" uv pip install --python "$VENV_PY" "${PINS[@]}"
  else
    info "newest versions"
    VIRTUAL_ENV="$VENV_DIR" uv pip install --python "$VENV_PY" -U mlx mlx-lm mlx-vlm transformers pillow
    echo "  ⚠️  --latest resolves mlx-vlm freely. 0.7.1 is the one release that"
    echo "      cannot run here (exact APC dies after the first short prompt,"
    echo "      upstream #2259); 0.7.2 and later are fine. A new release can"
    echo "      still move the APC planner -- see the PINS block in this script."
  fi
  ok "mlx-vlm $("$VENV_PY" -c 'import importlib.metadata as m;print(m.version("mlx-vlm"))')"
fi

# ── 6. Metal-Check ────────────────────────────────────────────────────────────
echo
echo "[6/8] Metal / MLX self-test"
if [[ -x "$VENV_PY" ]]; then
  "$VENV_PY" - <<'PY'
import mlx.core as mx

GiB = 1 << 30
a = mx.ones((512, 512), dtype=mx.float16)
mx.eval(a @ a)                       # forces a real Metal kernel execution
info = mx.device_info()
ws = info["max_recommended_working_set_size"] / GiB
ram = info["memory_size"] / GiB
print(f"  ✓ {info['device_name']} ({info['architecture']}), matmul on {mx.default_device()} ok")
print(f"  · RAM {ram:.0f} GiB, Metal working set {ws:.1f} GiB")
# 15.2 GiB of weights + 1.5 GiB reserve; below that practically nothing is left for KV.
if ws < 18:
    print("  ⚠️  Working set < 18 GiB -- model + reserve do not fit. Raise wired_limit!")
elif ws < 24 and ram >= 30:
    print("  ⚠️  Working set is the macOS default (2/3 of RAM). With")
    print("      sudo sysctl -w iogpu.wired_limit_mb=26624")
    print("      the context budget doubles. Details: README.md")
PY
else
  warn "skipped (no venv)"
fi

# ── 7. Patches ────────────────────────────────────────────────────────────────
echo
echo "[7/8] Patches (site-packages)"
if [[ -x "$BUNDLE_DIR/patches/apply-patches.sh" && -x "$VENV_PY" ]]; then
  if (( CHECK_ONLY == 1 )); then
    MLX_VENV_PY="$VENV_PY" "$BUNDLE_DIR/patches/apply-patches.sh" --check
  else
    MLX_VENV_PY="$VENV_PY" "$BUNDLE_DIR/patches/apply-patches.sh"
  fi
else
  warn "patches/apply-patches.sh not executable or venv missing"
fi

# ── 8. Modelle ────────────────────────────────────────────────────────────────
echo
echo "[8/8] Model weights"
have_model() { [[ -f "$1/config.json" ]] }
if (( SKIP_MODEL == 1 )); then
  info "skipped (--skip-model)"
elif (( CHECK_ONLY == 1 )); then
  have_model "$MODEL_DIR" && ok "model: $(du -shL "$MODEL_DIR" | cut -f1)" || warn "model missing: $MODEL_DIR"
  have_model "$DRAFT_DIR" && ok "drafter: $(du -shL "$DRAFT_DIR" | cut -f1)" || warn "drafter missing: $DRAFT_DIR"
else
  if have_model "$MODEL_DIR"; then
    ok "model present ($(du -shL "$MODEL_DIR" | cut -f1))"
  else
    info "downloading $MODEL_REPO (~15 GiB, resumable, Ctrl-C safe at any time)"
    "$BUNDLE_DIR/download-mlx-model.sh" "$MODEL_REPO" "$MODEL_DIR"
  fi
  if have_model "$DRAFT_DIR"; then
    ok "MTP drafter present ($(du -shL "$DRAFT_DIR" | cut -f1))"
  else
    info "downloading $DRAFT_REPO (~0.25 GiB) -- worth +58..132% decode"
    "$BUNDLE_DIR/download-mlx-model.sh" "$DRAFT_REPO" "$DRAFT_DIR"
  fi
  # Alias symlink: with mlx-vlm the request model name IS the load path.
  ln -sfn "$MODEL_DIR" "$MLX_MODELS/$MODEL_ALIAS"
  ok "alias symlink: $MLX_MODELS/$MODEL_ALIAS -> $MODEL_DIR"
fi

# ── Directories ───────────────────────────────────────────────────────────────
if (( CHECK_ONLY == 0 )); then
  mkdir -p "${STATE_DIR:-$HOME/.mlx-qwen38}/logs" "${STATE_DIR:-$HOME/.mlx-qwen38}/apc"
fi

echo
echo "──────────────────────────────────────────────────────────────"
echo "  Done. Next steps:"
echo
echo "  1) Raise the wired limit (MOST IMPORTANT step, needs sudo):"
echo "       sudo $BUNDLE_DIR/set-iogpu-wired-limit.sh          # ${RAM_GB} GB -> $WIRED_SUGGEST"
echo "     Persistent (survives reboots -- sysctl itself does NOT):"
echo "       sudo $BUNDLE_DIR/install-wired-limit-daemon.sh"
echo "     The daemon computes the value from hw.memsize at every boot -- no"
echo "     RAM-specific value is hardcoded anywhere. One call instead of four"
echo "     sudo lines: those broke when pasted and half-completed."
echo
echo "  2) Start the server:"
echo "       $BUNDLE_DIR/start-mlx_qwen3.8.sh"
echo "     The start script prints this machine's computed context budget."
echo
echo "  3) Configure the client to the budget (context_length,"
echo "     max_tokens=16384) -- values and rationale in README.md."
echo "──────────────────────────────────────────────────────────────"
