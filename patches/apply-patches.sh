#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm: lokale Patches auf site-packages anwenden.
#
# Die Patches leben in site-packages und verschwinden bei JEDEM
# `uv pip install`/`pip install -U` von mlx-vlm. Dieses Skript stellt sie wieder
# her. Idempotent (Reverse-Dry-Run als Test).
#
#   ./apply-patches.sh            anwenden
#   ./apply-patches.sh --check    nur Status zeigen
#   ./apply-patches.sh --revert   zuruecknehmen
#
# venv-Python per Env:  MLX_VENV_PY=/pfad/zu/.venv/bin/python ./apply-patches.sh
#
# ── ENTHALTENE PATCHES ───────────────────────────────────────────────────────
#
# 0002-pr1901-apc-short-prompt.patch   (Upstream-PR #1901, GEMERGED 2026-08-15)
#   "Stop a short first prompt from disabling prefix caching" — gemerged DREI
#   TAGE NACH dem 0.6.13-Release, also nicht im PyPI-Stand von 0.6.13.
#   Bug: ein Prompt kuerzer als APC_EXACT_PREFIX_GUARD_TOKENS (16) macht die
#   Checkpoint-Laenge negativ; auf 1 geklammert wird ein EIN-TOKEN-Snapshot
#   gespeichert, der auf den Anfang jedes spaeteren Prompts passt. Da ein neuer
#   Checkpoint nur entsteht, wenn nichts wiederverwendet wurde, speichert dieser
#   Tenant nie wieder einen brauchbaren. Signatur im Log: cached_tokens=1 bei
#   grossen Prompts.
#   GEMESSEN (M5 Pro, 48 GB): 13 Requests mit cached_tokens==1 = 1055 s
#   verlorene Prefill-Zeit, ausgeloest von nur 6 kurzen Prompts. Nach dem Patch:
#   kurzer Prompt, dann 35881-Token-Prompt zweimal → Lauf 2 cached=35865,
#   Prefill 318 ms (vorher cached=1, 94 s).
#   Auf einer 32-GB-/M5-Basis-Maschine wiegt das MEHR, nicht weniger: der
#   Prefill ist dort rechenlimitiert und langsamer.
#   ENTFERNEN, sobald mlx-vlm > 0.6.13 installiert ist — dann meldet dieses
#   Skript "KONFLIKT", was hier "upstream schon drin" heisst.
#
# 0010-qwen38-apc-single-snapshot.patch   (LOKAL, kein Upstream-PR)
#   Unterdrueckt den redundanten Voll-Snapshot pro Request. mlx-vlm legt sonst
#   ZWEI fast identische Snapshots ab: den Checkpoint bei len-16 (Guard) und den
#   vollen Prompt. Getroffen wird gemessen immer der Checkpoint (prompt 3194 →
#   cached 3178, Differenz genau 16), der Voll-Snapshot ist toter Ballast und
#   halbiert die Zahl der warm gehaltenen Konversationen.
#   GEMESSEN (APC_EXACT_CACHE_ENTRIES=2, M5 Pro):
#     ohne Patch: "Turn 1 wiederholt" cached=0     (Voll-Snapshot verdraengt alles)
#     mit  Patch: "Turn 1 wiederholt" cached=3178, Prefill 6737 → 219 ms
#   Wird ueber QWEN38_APC_SINGLE_SNAPSHOT=1 aktiviert (setzt das Start-Skript).
#   OHNE die Env-Variable ist der Patch inert = exaktes Upstream-Verhalten;
#   das ist der Rollback-Pfad.
#
# NICHT ENTHALTEN — bewusst:
#   Der KV-Window-Patch (QWEN38_KV_WINDOW, Gleitfenster auf den 16 Full-Attn-
#   Layern) aus der 48-GB-Maschine. Er halbiert zwar genau die Groesse, die auf
#   32 GB weh tut, wurde aber am 2026-08-17 mit needle_hybrid.py WIDERLEGT: eine
#   Nadel ausserhalb des Fensters ging verloren und wurde im 40-%-Fall sogar
#   halluziniert (8347 statt 8342). Speicher spart man hier ueber KV_BITS=8 und
#   kleineren Kontext, nicht ueber Fensterung.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PATCH_DIR="${0:A:h}"
VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"

[[ -x "$VENV_PY" ]] || { echo "ERROR: venv-Python nicht gefunden: $VENV_PY"; exit 1; }
SITE_PACKAGES=$("$VENV_PY" -c "import mlx_vlm, os; print(os.path.dirname(os.path.dirname(mlx_vlm.__file__)))" 2>/dev/null) \
  || { echo "ERROR: mlx_vlm ist in $VENV_PY nicht importierbar."; exit 1; }

MODE="apply"
[[ "${1:-}" == "--revert" ]] && MODE="revert"
[[ "${1:-}" == "--check" ]]  && MODE="check"

echo "  site-packages: $SITE_PACKAGES"
echo "  mlx-vlm      : $("$VENV_PY" -c 'import importlib.metadata as m;print(m.version("mlx-vlm"))')"

patches=( "$PATCH_DIR"/*.patch(N) )
[[ ${#patches[@]} -gt 0 ]] || { echo "  Keine .patch-Dateien in $PATCH_DIR"; exit 0; }

for p in "${patches[@]}"; do
  name="${p:t}"

  # Bereits angewendet? Der Reverse-Dry-Run ist der zuverlaessige Test: er
  # gelingt genau dann, wenn der Patch drin ist.
  if patch -R -p1 --dry-run --force -d "$SITE_PACKAGES" < "$p" &>/dev/null; then
    applied=1
  else
    applied=0
  fi

  case "$MODE" in
    check)
      echo "  $([[ $applied == 1 ]] && echo '[angewendet]' || echo '[  offen   ]')  $name"
      continue
      ;;
    revert)
      if [[ $applied == 1 ]]; then
        patch -R -p1 -d "$SITE_PACKAGES" < "$p" >/dev/null
        echo "  zurueckgenommen: $name"
      else
        echo "  nicht angewendet, uebersprungen: $name"
      fi
      continue
      ;;
  esac

  if [[ $applied == 1 ]]; then
    echo "  bereits angewendet: $name"
    continue
  fi

  if ! patch -p1 --dry-run -d "$SITE_PACKAGES" < "$p" &>/dev/null; then
    echo "  ⚠️  KONFLIKT: $name laesst sich nicht anwenden."
    echo "      Bei 0002 heisst das in der Regel: upstream gemerged (mlx-vlm > 0.6.13)"
    echo "      → Datei aus $PATCH_DIR entfernen."
    echo "      Bei 0010 heisst es: ar.py hat sich geaendert → Patch neu schreiben."
    continue
  fi

  patch -p1 -d "$SITE_PACKAGES" < "$p" >/dev/null
  echo "  angewendet: $name"
done

[[ "$MODE" == "check" ]] && exit 0

# Ein Patch, der gegen eine veraenderte Upstream-Version nur teilweise greift,
# faellt hier auf.
if ! "$VENV_PY" -c "import mlx_vlm.apc, mlx_vlm.generate.ar" 2>/dev/null; then
  echo "  ⚠️  WARNUNG: mlx_vlm laesst sich nach dem Patchen NICHT importieren."
  echo "      Zuruecknehmen mit:  $0 --revert"
  exit 1
fi
echo "  mlx_vlm importierbar — ok."
