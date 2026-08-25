#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Prerequisites fuer Qwen3.8-27B auf MLX  –  Apple Silicon ab 32 GB
#
# Richtet von einem frischen macOS aus alles ein, was start-mlx_qwen3.8.sh
# braucht: Xcode CLT → uv → venv (Python 3.12) → mlx-vlm + Patches → Modell +
# MTP-Drafter → Verzeichnisse. Idempotent: erneut ausfuehrbar, bereits erledigte
# Schritte werden uebersprungen, Downloads setzen fort.
#
# Verwendung:
#   ./install-prereqs.sh                 # alles, mit gepinnten Versionen
#   ./install-prereqs.sh --skip-model    # nur Software, kein 15-GB-Download
#   ./install-prereqs.sh --latest        # neueste Versionen statt der gepinnten
#   ./install-prereqs.sh --check         # nur pruefen, nichts aendern
#
# Env-Overrides:  MLX_HOME (Default ~/src/mlx), MLX_MODELS, PYTHON_VERSION
#
# WAS DAS SKRIPT NICHT TUT: sudo. Das Wired-Limit (iogpu.wired_limit_mb) ist der
# wichtigste Tuning-Schritt, aendert aber Systemzustand — die noetigen
# Befehle werden am Ende nur AUSGEGEBEN, s. auch README.md.
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
    *) echo "Unbekannte Option: $a"; exit 1 ;;
  esac
done

# Gepinnter, auf einem M5 Pro / macOS 26 als funktionierend VERIFIZIERTER Stand
# (2026-08-19). mlx-vlm 0.6.15 ist die Version, gegen die der Patch in
# patches/ geschrieben ist. APC ist ab 0.6.13 upstream korrekt; der
# Kurzprompt-Fix (PR #1901) ist ab 0.6.14 enthalten.
# Mit --latest bekommt man Neueres; dann kann apply-patches.sh "KONFLIKT" melden
# (heisst: upstream gemerged → Patch loeschen) und die Messwerte im Start-Skript
# gelten nicht mehr unbesehen.
PINS=(
  "mlx==0.32.1"
  "mlx-lm==0.31.3"
  "mlx-vlm==0.6.15"
  "transformers==5.15.0"
  "numpy==2.5.2"
  "huggingface-hub==1.27.0"
  "pillow==12.3.0"
)

MODEL_REPO="mlx-community/Qwen3.8-27B-4bit"
MODEL_DIR="$MLX_MODELS/Qwen3.8-27B-MLX-4bit"
DRAFT_REPO="mlx-community/Qwen3.8-27B-MTP-4bit"
DRAFT_DIR="$MLX_MODELS/Qwen3.8-27B-MTP-4bit"
MODEL_ALIAS="${MODEL_ALIAS:-Qwen3.8-27B-local}"
# Downloadgroesse: Modell ~15,0 GiB (3 Shards + Tokenizer), Drafter ~0,25 GiB.
NEEDED_GB=20

ok()   { echo "  ✓ $*" }
info() { echo "  · $*" }
warn() { echo "  ⚠️  $*" >&2 }
die()  { echo "  ✗ $*" >&2; exit 1 }

echo "──────────────────────────────────────────────────────────────"
echo "  Qwen3.8-27B / MLX  —  Setup fuer Apple Silicon (32 / 48 GB)"
echo "  Ziel-venv : $VENV_DIR"
echo "  Modelle   : $MLX_MODELS"
echo "──────────────────────────────────────────────────────────────"

# ── 1. Hardware/OS ────────────────────────────────────────────────────────────
echo
echo "[1/8] Hardware & macOS"
[[ "$(uname -s)" == "Darwin" ]] || die "Kein macOS."
[[ "$(uname -m)" == "arm64" ]]  || die "Kein Apple Silicon (uname -m = $(uname -m)). MLX braucht arm64."
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "?")
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
ok "$CHIP · ${RAM_GB} GB RAM · macOS $(sw_vers -productVersion)"
# Empfehlung fuers Wired-Limit: RAM minus Reserve fuer macOS, absolut gerechnet
# (6 GiB bis 32 GB RAM, 8 GiB darueber). Dieselbe Regel wie in
# set-iogpu-wired-limit.sh — dort steht die Begruendung.
#
# ACHTUNG, HIER STAND EIN GEFAEHRLICHER WERT: bis 2026-08-24 schlug dieser
# Zweig fuer RAM >= 44 GB die 45056 vor. Das sind auf einer 48-GB-Maschine nur
# 4 GiB Reserve, und genau an diesem Wert ist die Testmaschine am 2026-08-21 in
# eine Kernel-Panik gelaufen (watchdog timeout, s. README). Zusaetzlich
# widersprach der Vorschlag der plist, die 26624 setzte — zwei verschiedene
# falsche Werte in einer Anleitung. Beides rechnet jetzt dasselbe Skript.
if (( RAM_GB <= 32 )); then
  WIRED_SUGGEST=$(( RAM_GB * 1024 - 6144 ))
else
  WIRED_SUGGEST=$(( RAM_GB * 1024 - 8192 ))
fi
# Unter dem macOS-Default (2/3 des RAM) waere der Eingriff eine VERSCHLECHTERUNG.
# Trifft nur sehr kleine Maschinen (16 GB: 10240 < 10922), die laut Warnung
# unten ohnehin nicht tragen. set-iogpu-wired-limit.sh verweigert solche Werte —
# hier gaebe es sonst einen Vorschlag, den das Skript danach ablehnt.
_WIRED_FLOOR=$(( RAM_GB * 1024 * 2 / 3 ))
(( WIRED_SUGGEST < _WIRED_FLOOR )) && WIRED_SUGGEST=$_WIRED_FLOOR
if   (( RAM_GB >= 44 )); then PROFILE_HINT="roomy"
elif (( RAM_GB >= 32 )); then PROFILE_HINT="balanced"
else                          PROFILE_HINT="lean"
fi
if (( RAM_GB < 32 )); then
  warn "Nur ${RAM_GB} GB RAM. Die Gewichte allein belegen 15,2 GiB — unter 32 GB"
  warn "bleibt kein brauchbarer Kontext. Kleineres Modell/Quant waehlen."
elif (( RAM_GB == 32 )); then
  info "32 GB: knapp, aber tragfaehig — Profil '$PROFILE_HINT'. S. README.md."
else
  info "Profil '$PROFILE_HINT' passt zu dieser Maschine (PROFILE=auto waehlt es selbst)."
fi
FREE_GB=$(( $(df -k "$HOME" | awk 'NR==2{print $4}') / 1048576 ))
info "Freier Plattenplatz: ${FREE_GB} GB (gebraucht: ~${NEEDED_GB} GB fuer Modell+Drafter,"
info "dazu bis zu 40 GB fuer den APC-SSD-Cache — der Deckel steht im Start-Skript)"
(( SKIP_MODEL == 1 || FREE_GB > NEEDED_GB )) || die "Zu wenig Plattenplatz."

# ── 2. Xcode Command Line Tools ───────────────────────────────────────────────
echo
echo "[2/8] Xcode Command Line Tools"
if xcode-select -p &>/dev/null; then
  ok "vorhanden ($(xcode-select -p))"
else
  warn "fehlen. Metal-Toolchain wird gebraucht. Installieren mit:"
  echo "      xcode-select --install"
  (( CHECK_ONLY == 1 )) || die "Erst CLT installieren, dann dieses Skript erneut."
fi

# ── 3. uv ─────────────────────────────────────────────────────────────────────
echo
echo "[3/8] uv (Paket-/venv-Manager)"
if command -v uv &>/dev/null; then
  ok "uv $(uv --version | awk '{print $2}')"
elif (( CHECK_ONLY == 1 )); then
  warn "uv fehlt"
else
  info "installiere uv nach ~/.local/bin (offizieller Installer von astral.sh)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv &>/dev/null || die "uv-Installation fehlgeschlagen. Alternativ: brew install uv"
  ok "uv $(uv --version | awk '{print $2}')"
  info "PATH-Eintrag fuer neue Shells:  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# ── 4. venv ───────────────────────────────────────────────────────────────────
echo
echo "[4/8] Virtualenv (Python $PYTHON_VERSION)"
if (( CHECK_ONLY == 1 )); then
  [[ -x "$VENV_PY" ]] && ok "$($VENV_PY -V)" || warn "venv fehlt: $VENV_DIR"
else
  mkdir -p "$MLX_HOME" "$MLX_MODELS"
  if [[ -x "$VENV_PY" ]]; then
    ok "vorhanden: $($VENV_PY -V)"
  else
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    ok "angelegt: $($VENV_PY -V)"
  fi
fi

# ── 5. Pakete ─────────────────────────────────────────────────────────────────
echo
echo "[5/8] mlx-vlm & Abhaengigkeiten"
if (( CHECK_ONLY == 1 )); then
  [[ -x "$VENV_PY" ]] && "$VENV_PY" - <<'PY' || warn "venv fehlt"
import importlib.metadata as m
for p in ("mlx", "mlx-lm", "mlx-vlm", "transformers", "numpy", "huggingface-hub", "pillow"):
    try:
        print(f"  · {p:18s} {m.version(p)}")
    except Exception:
        print(f"  ⚠️  {p:18s} FEHLT")
PY
else
  if (( PINNED == 1 )); then
    info "gepinnter Stand (--latest fuer neueste Versionen)"
    VIRTUAL_ENV="$VENV_DIR" uv pip install --python "$VENV_PY" "${PINS[@]}"
  else
    info "neueste Versionen"
    VIRTUAL_ENV="$VENV_DIR" uv pip install --python "$VENV_PY" -U mlx mlx-lm mlx-vlm transformers pillow
  fi
  ok "mlx-vlm $("$VENV_PY" -c 'import importlib.metadata as m;print(m.version("mlx-vlm"))')"
fi

# ── 6. Metal-Check ────────────────────────────────────────────────────────────
echo
echo "[6/8] Metal / MLX-Selbsttest"
if [[ -x "$VENV_PY" ]]; then
  "$VENV_PY" - <<'PY'
import mlx.core as mx

GiB = 1 << 30
a = mx.ones((512, 512), dtype=mx.float16)
mx.eval(a @ a)                       # zwingt eine echte Metal-Kernel-Ausfuehrung
info = mx.device_info()
ws = info["max_recommended_working_set_size"] / GiB
ram = info["memory_size"] / GiB
print(f"  ✓ {info['device_name']} ({info['architecture']}), Matmul auf {mx.default_device()} ok")
print(f"  · RAM {ram:.0f} GiB, Metal-Working-Set {ws:.1f} GiB")
# 15,2 GiB Gewichte + 1,5 GiB Reserve; darunter bleibt fuer KV praktisch nichts.
if ws < 18:
    print("  ⚠️  Working-Set < 18 GiB — Modell + Reserve passen nicht. wired_limit anheben!")
elif ws < 24 and ram >= 30:
    print("  ⚠️  Working-Set ist der macOS-Default (2/3 RAM). Mit")
    print("      sudo sysctl -w iogpu.wired_limit_mb=26624")
    print("      verdoppelt sich das Kontext-Budget. Details: README.md")
PY
else
  warn "uebersprungen (kein venv)"
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
  warn "patches/apply-patches.sh nicht ausfuehrbar oder venv fehlt"
fi

# ── 8. Modelle ────────────────────────────────────────────────────────────────
echo
echo "[8/8] Modellgewichte"
have_model() { [[ -f "$1/config.json" ]] }
if (( SKIP_MODEL == 1 )); then
  info "uebersprungen (--skip-model)"
elif (( CHECK_ONLY == 1 )); then
  have_model "$MODEL_DIR" && ok "Modell: $(du -shL "$MODEL_DIR" | cut -f1)" || warn "Modell fehlt: $MODEL_DIR"
  have_model "$DRAFT_DIR" && ok "Drafter: $(du -shL "$DRAFT_DIR" | cut -f1)" || warn "Drafter fehlt: $DRAFT_DIR"
else
  if have_model "$MODEL_DIR"; then
    ok "Modell vorhanden ($(du -shL "$MODEL_DIR" | cut -f1))"
  else
    info "lade $MODEL_REPO (~15 GiB, resume-faehig, Ctrl-C jederzeit gefahrlos)"
    "$BUNDLE_DIR/download-mlx-model.sh" "$MODEL_REPO" "$MODEL_DIR"
  fi
  if have_model "$DRAFT_DIR"; then
    ok "MTP-Drafter vorhanden ($(du -shL "$DRAFT_DIR" | cut -f1))"
  else
    info "lade $DRAFT_REPO (~0,25 GiB) — bringt +58..132 % Decode"
    "$BUNDLE_DIR/download-mlx-model.sh" "$DRAFT_REPO" "$DRAFT_DIR"
  fi
  # Alias-Symlink: bei mlx-vlm IST der Request-Modellname der Ladepfad.
  ln -sfn "$MODEL_DIR" "$MLX_MODELS/$MODEL_ALIAS"
  ok "Alias-Symlink: $MLX_MODELS/$MODEL_ALIAS → $MODEL_DIR"
fi

# ── Verzeichnisse ─────────────────────────────────────────────────────────────
if (( CHECK_ONLY == 0 )); then
  mkdir -p "${STATE_DIR:-$HOME/.mlx-qwen38}/logs" "${STATE_DIR:-$HOME/.mlx-qwen38}/apc"
fi

echo
echo "──────────────────────────────────────────────────────────────"
echo "  Fertig. Naechste Schritte:"
echo
echo "  1) Wired-Limit anheben (WICHTIGSTER Schritt, braucht sudo):"
echo "       sudo $BUNDLE_DIR/set-iogpu-wired-limit.sh          # ${RAM_GB} GB -> $WIRED_SUGGEST"
echo "     Persistent (ueberlebt Neustarts — sysctl selbst tut das NICHT):"
echo "       sudo $BUNDLE_DIR/install-wired-limit-daemon.sh"
echo "     Der Daemon rechnet den Wert bei jedem Boot aus hw.memsize — kein"
echo "     RAM-spezifischer Wert liegt irgendwo fest. Ein Aufruf statt vier"
echo "     sudo-Zeilen: die brachen beim Kopieren um und liefen halb durch."
echo
echo "  2) Server starten:"
echo "       $BUNDLE_DIR/start-mlx_qwen3.8.sh"
echo "     Das Start-Skript druckt das errechnete Kontext-Budget dieser Maschine."
echo
echo "  3) Den Client auf das Budget einstellen (context_length,"
echo "     max_tokens=8192) — Werte und Begruendung in README.md."
echo "──────────────────────────────────────────────────────────────"
