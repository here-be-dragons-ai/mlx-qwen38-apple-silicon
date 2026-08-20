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
# 0020-dflash2-qwen38.patch   (LOKAL, Upstream-PR-Kandidat)
#   Ruestet DFlash 2 im Drafter qwen3_dflash nach. mlx-vlm implementiert bis
#   einschliesslich 0.6.15 (und auf main) nur DFlash v1; fuer Qwen3.8-27B gibt
#   es aber ausschliesslich einen v2-Drafter. Ergaenzt werden:
#     GroupedDynamicCausalConv  Two-Tap-Conv mit statischem + eingabeabhaengigem
#                               Kernel, pro Layer vor Attention und MLP
#     CandidateSelector         Top-k je Blockposition + Pfadwahl ueber eine
#                               niedrigrangige bilineare Form
#   Transkribiert aus der MLX-Referenz z-lab/dflash (dflash/model_mlx.py).
#   VERIFIZIERT: 81/81 Parameter stimmen in Name und Form mit dem Checkpoint
#   ueberein, Conv max|diff| = 0 gegen die Referenz, Selector-Pfade identisch,
#   Ausgabe bei temperature 0 bit-identisch zum MTP-Drafter (4/4).
#   GEMESSEN (M5 Pro, block_size 4): Decode +13..24 % gegenueber MTP.
#   Blocksweep: 3 -> +6 %, 4 -> +19 %, 5 -> +20 %, 8 -> +6 % (der Checkpoint ist
#   auf 8 ausgelegt, das ist auf einem 4bit-Target die schlechteste Wahl).
#   ABER: unter draft_kind=dflash greift APC gar nicht (cached_tokens=0 in jedem
#   Turn, APC_TRACE zeigt keinen einzigen Lookup). Das liegt an der
#   dflash-Integration in mlx-vlm, nicht am Patch — MTP trifft mit angewandtem
#   Patch weiter. Deshalb ist DRAFT_KIND=mtp Default; s. README.
#   Ohne DRAFT_KIND=dflash ist der Patch inert.
#
# 0021-speculative-apc-routing.patch   (LOKAL, Upstream-PR-Kandidat)
#   Macht den Prefix-Cache fuer Nicht-MTP-Drafter ueberhaupt erst erreichbar.
#   server/generation.py routet jeden Drafter ausser mtp in eine zweite
#   Generierungsschleife (_run_speculative), die ihren eigenen Prompt-Cache baut
#   und den apc_manager NIE verdrahtet — Folge: cached_tokens=0 in jedem Request,
#   APC_TRACE zeigt keinen einzigen Lookup. Der Continuous-Batching-Pfad kann
#   dflash laengst (generisch ueber draft_kind, bekommt apc_manager, draft_kind
#   und draft_block_size in derselben Zeile); nur die Weiche hielt ihn fern.
#   Der Patch macht den Batch-Pfad ueber MLX_VLM_SPECULATIVE_BATCH=1 erreichbar,
#   Default unveraendert. Das Start-Skript setzt die Variable bei DRAFT_KIND!=mtp.
#   GEMESSEN (5,8k-Konversation, Turn 2): cached 0 -> 5748/5788. Decode
#   unveraendert (im Mittel 40,8 statt 41,5 t/s), beim 5767-Token-Prompt besser
#   (38,4 -> 40,9 t/s). --draft-block-size wirkt weiter, MAX_NUM_SEQS=2 laeuft,
#   MTP unveraendert (cached 5772).
#
# ── FREMDE, NOCH OFFENE UPSTREAM-PRs (cherry-gepickt) ────────────────────────
# Beide sind Bugfixes anderer Leute, die upstream noch offen sind. Sobald sie
# gemerged sind, meldet dieses Skript "KONFLIKT" — dann entfernen.
#
# 0030-pr1956-speculative-quantized-kv.patch   (PR #1956, @Codcore, offen)
#   "Fix speculative decoding against a quantized KV cache".
#   SELBST REPRODUZIERT: mit KV_BITS=8 und MAX_NUM_SEQS=2 sterben zwei
#   parallele Requests mit HTTP 500 und
#     AttributeError: 'tuple' object has no attribute 'shape'
#   Der Verify-Pfad nimmt an, keys sei EIN Array; ein quantisierter Cache liefert
#   ein Tupel. Mit Patch laufen dieselben zwei Requests korrekt durch.
#   BETRIFFT UNS bei MAX_NUM_SEQS > 1: das lean-Profil setzt KV_BITS=8. Mit
#   MAX_NUM_SEQS=1 (unser Default in allen Profilen) tritt der Fehler NICHT auf —
#   auch nicht bei 15838 Token und Quantisierung ab Token 0, extra geprueft.
#
# 0031-pr1835-recurrent-cache-no-trim.patch    (PR #1835, @kylesyx, offen)
#   "Decline prefix-cache reuse for non-trimmable recurrent caches".
#   _prefix_cache_trim_amount() prueft nur, ob der Praefix noch VORHANDEN ist,
#   nicht ob der Cache ueberhaupt trimmbar ist. Die ArraysCache der 48
#   GDN-Layer von Qwen3.8 ist beides nicht — sie faellt durch die Pruefung und
#   der Aufrufer stirbt an c.trim(n_drop).
#   SELBST REPRODUZIERT auf Einheitsebene mit den echten Cache-Klassen:
#     ohne Patch  _prefix_cache_trim_amount([ArraysCache, KVCache], 10) = 10
#                 -> AttributeError: 'ArraysCache' object has no attribute 'trim'
#     mit  Patch  = None (Wiederverwendung abgelehnt), reines KVCache-Modell
#                 liefert weiter 10 — keine Regression fuer Attention-Modelle.
#   BETRIFFT UNSEREN SERVER NICHT: _prefix_cache_trim_amount wird nur aus
#   dispatch.stream_generate aufgerufen, der Serverpfad geht da nicht durch.
#   Drin als Vorsorge fuer mlx_vlm.chat_ui, die generate-CLI und eigene Skripte,
#   die prompt_cache_state durchreichen.
#
# ── ERLEDIGT / OBSOLET ───────────────────────────────────────────────────────
#
# 0002-pr1901-apc-short-prompt.patch   ENTFERNT 2026-08-19 beim Upgrade auf
#   mlx-vlm 0.6.15. Der Upstream-PR #1901 wurde am 2026-08-15 gemerged, also
#   drei Tage NACH dem 0.6.13-Release — deshalb war er in 0.6.13 noch noetig
#   und ist ab 0.6.14 enthalten. Gegengeprueft per Reverse-Dry-Run gegen den
#   0.6.15-Wheel: beide Hunks (apc.py, server/app.py) sind drin.
#   Was er behob: ein Prompt kuerzer als APC_EXACT_PREFIX_GUARD_TOKENS (16)
#   machte die Checkpoint-Laenge negativ; auf 1 geklammert wurde ein
#   EIN-TOKEN-Snapshot gespeichert, der auf den Anfang jedes spaeteren Prompts
#   passte. Da ein neuer Checkpoint nur entsteht, wenn nichts wiederverwendet
#   wurde, speicherte dieser Tenant nie wieder einen brauchbaren.
#   Signatur im Log: cached_tokens=1 bei grossen Prompts.
#   GEMESSEN (M5 Pro, 48 GB): 13 Requests mit cached_tokens==1 = 1055 s
#   verlorene Prefill-Zeit, ausgeloest von nur 6 kurzen Prompts.
#   WER AUF 0.6.13 ZURUECKGEHT, braucht ihn wieder — er liegt in der
#   Git-Historie dieses Repos.
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
    echo "      Heisst in der Regel: ar.py hat sich upstream geaendert."
    echo "      Pruefen, ob der Effekt (Einzel-Snapshot) inzwischen upstream ist —"
    echo "      wenn ja, Datei aus $PATCH_DIR entfernen, sonst Patch neu schreiben."
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
