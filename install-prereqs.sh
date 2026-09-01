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

# Pinned state, VERIFIED as working on an M5 Pro / macOS 26 (2026-08-25).
# mlx-vlm 0.6.16 is the version the patches in patches/ are written against.
# 0.6.16 matters for three reasons: DFlash 2 ships upstream (PR #2014, which
# made local patch 0040 obsolete), the ArraysCache buffer leak that killed
# generations at ~10.3k tokens is fixed (#1972 via PR #1984), and Qwen3.8-27B
# support landed (#1899).
# mlx 0.32.2 is on PyPI since 2026-08-25, including mlx-metal and
# macosx_26_0_arm64 wheels -- the source build documented in docs/build-mlx.md
# is no longer needed. Patch 0013 is still required (the default dispatch still
# does not route to force_fused) and applies unchanged.
# --latest gets you something newer; apply-patches.sh may then report "CONFLICT"
# (meaning: merged upstream -> delete the patch) and the measured values in the
# start script no longer hold unexamined.
PINS=(
  "mlx==0.32.2"
  "mlx-lm==0.31.3"
  "mlx-vlm==0.7.0rc0"
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
for p in ("mlx", "mlx-lm", "mlx-vlm", "transformers", "numpy", "huggingface-hub", "pillow"):
    try:
        print(f"  · {p:18s} {m.version(p)}")
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
echo "     max_tokens=8192) -- values and rationale in README.md."
echo "──────────────────────────────────────────────────────────────"
