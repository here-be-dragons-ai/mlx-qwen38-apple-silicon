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
# 0013-force-fused-sdpa-head-dim-256.patch   (LOKAL, kein Upstream-PR)
#   Qwen3.8 hat head_dim 256. mlx' Default-Dispatch laesst fused Full-Attention
#   nur fuer head_dim 64/80/128 zu — die 16 Full-Attn-Layer laufen deshalb auf
#   dem unfused Graph und materialisieren pro Layer einen Score-Transienten von
#   O(n_heads x qL x kL). Das ist der eigentliche Grund, warum PREFILL_STEP bei
#   uns ein RAM-Hebel ist.
#   mlx 0.32.2 (PR #4185) stellt die 192/256-Kernel wieder her, erreichbar NUR
#   ueber force_fused=True; der Default-Dispatch routet weiterhin nicht dorthin.
#   Der PR begruendet das ausdruecklich damit, dass nur die Runtime ihr
#   Speicherbudget kennt — das trifft hier zu.
#   Eng gefasst: nur qL > 1 (Prefill/Verify, nicht Decode), nur head_dim
#   192/256, nur ohne Array-Maske und ohne Sinks. Wirft force_fused einmal, wird
#   der Pfad dauerhaft abgeschaltet und einmal geloggt.
#   INERT AUF mlx < 0.32.2: die Probe beim Import faellt auf TypeError.
#   VERIFIZIERT auf mlx 0.32.0: _FORCE_FUSED == False, Verhalten unveraendert.
#   Rollback: QWEN38_FORCE_FUSED_SDPA=0
#   ACHTUNG: PR #3842 (fused head_dim 256 auf NAX/M5) verlangt qL >= 1024 —
#   PROFILE=lean setzt PREFILL_STEP=512 und faellt damit heraus.
#
# 0014-quantized-kv-start-uniform.patch   (LOKAL, kein Upstream-PR)
#   quantized_kv_start galt auf dem Batch-Pfad nur fuer TurboQuant. Auf dem
#   uniform-Pfad — also --kv-bits ohne --kv-quant-scheme turboquant, unser
#   Default — wurde ab Token 0 quantisiert, egal was --quantized-kv-start sagt.
#   GEMESSEN mit _make_cache(kv_bits=8, quantized_kv_start=8192):
#     ohne Patch  prefill_length=1000  -> BatchQuantizedKVCache  (falsch)
#     mit  Patch  prefill_length=1000  -> BatchKVCache           (f16)
#                 prefill_length=20000 -> BatchQuantizedKVCache
#   BETRIFFT PROFILE=lean im Normalbetrieb: dort ist KV_BITS=8 Default, und
#   seit DFlash 2 Default ist, setzt das Start-Skript MLX_VLM_SPECULATIVE_BATCH=1
#   — der Batch-Pfad laeuft also nicht mehr nur bei MAX_NUM_SEQS > 1.
#   Rollback: QUANT_KV_START=0
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
#   STATUS UPSTREAM: das zugehoerige Issue #1966 wurde am 2026-08-20 GESCHLOSSEN
#   — zugunsten von PR #1923 ("conservative DFlash APC prefix reuse", nur B=1,
#   text-only, exact-prefix). Dieser Patch landet also nicht in dieser Form; die
#   Abhaengigkeit bleibt bestehen, bis #1923 gemerged ist.
#
# 0041-dflash2-guard-invalid-bonus-token.patch   (LOKAL, kein Upstream-PR)
#   Nachfolger von 0022, auf den Upstream-Code aus 0040 umgezogen. DFlash 2 baut
#   den naechsten Block in DFlash2DraftModel.propose_block, nicht mehr in
#   DFlashDraftModel.draft_block — die v1-Methode ist fuer diesen Drafter tot.
#   Inhalt unveraendert: mx.array() wirft fuer Werte ausserhalb des int64-Bereichs
#   nur "RuntimeError: std::bad_cast", ohne Wert, ohne Index (reproduzierbar mit
#   mx.array([[2**63]], dtype=mx.int32)). Genau so starb am 2026-08-20 10:07 ein
#   Request nach 250 Tokens. Der Guard prueft gegen vocab_size und nennt den Wert.
#   Bewusst kein Clamping: ein still ersetztes Token verfaelscht die Ausgabe.
#   GEPRUEFT: PR #1959 hat diesen Guard NICHT — die Stelle ist upstream offen.
#
# ── FREMDE, NOCH OFFENE UPSTREAM-PRs (cherry-gepickt) ────────────────────────
# Bugfixes anderer Leute, die upstream noch offen sind. Sobald sie gemerged
# sind, meldet dieses Skript "KONFLIKT" — dann entfernen.
#
# 0040-pr1959-dflash2.patch   (PR #1959, offen, draft)
#   "Add DFlash 2 speculative decoding" — die Upstream-Fassung dessen, was hier
#   bis 2026-08-20 als lokaler Patch 0020 lag. ERSETZT 0020 vollstaendig.
#   Bringt gegenueber 0020 drei Dinge, die die eigene Transkription nicht hatte:
#     - dedizierter bit-exakter 4bit-M=4-Metal-Verifier-Kernel: streamt die vier
#       Verify-Zeilen zusammen, reused packed weights ueber alle vier Token
#     - distribution-preserving rejection sampling fuer temperature > 0
#       (0020 war nur gegen greedy auf Bit-Gleichheit geprueft)
#     - optionale In-Memory-Quantisierung des Drafters (MLX_VLM_DRAFT_BITS)
#   Upstream-Messung (M3 Ultra, BF16-Drafter, block 4): 31,85 -> 47,07 t/s
#   (1,48x), 500/500 Token identisch zum Lauf ohne Drafter, Acceptance 60,5 %.
#   VERIFIZIERT HIER: laesst sich sauber auf v0.6.15 anwenden (kein Rebase auf
#   #1899 noetig) und kollidiert mit keinem der lokalen Patches. Der vorhandene
#   Checkpoint Qwen3.8-27B-DFlash2-4bit laedt unveraendert (DFlash2DraftModel,
#   179 Parameter, 1,008 GiB) — keine Neukonvertierung noetig.
#   Der Codebook-Rename (candidate_selector.{predecessor,successor}_codebook ->
#   ...weight) sitzt upstream in DFlash2DraftModel.sanitize und ist identisch zu
#   dem, was 0020 hatte und was z-lab in dflash/model_mlx.py (e128a7e) macht.
#   MUSS ALS LETZTER PATCH LAUFEN: gegen den Stand MIT 0010..0031 erzeugt.
#
# 0030-pr1956-speculative-quantized-kv.patch   (PR #1956, @Codcore, offen)
#   "Fix speculative decoding against a quantized KV cache".
#   SELBST REPRODUZIERT: mit KV_BITS=8 und MAX_NUM_SEQS=2 sterben zwei
#   parallele Requests mit HTTP 500 und
#     AttributeError: 'tuple' object has no attribute 'shape'
#   Der Verify-Pfad nimmt an, keys sei EIN Array; ein quantisierter Cache liefert
#   ein Tupel. Mit Patch laufen dieselben zwei Requests korrekt durch.
#   EINORDNUNG KORRIGIERT AM 2026-08-20 — vorher stand hier, der Patch sei nur
#   bei MAX_NUM_SEQS > 1 relevant. Das galt, solange MTP Default war. Seit
#   DFlash 2 Default ist, setzt das Start-Skript MLX_VLM_SPECULATIVE_BATCH=1,
#   und _make_cache baut den Batch-Cache auch bei MAX_NUM_SEQS=1, sobald
#   KV_BITS gesetzt ist (generate/ar.py:796). Auf PROFILE=lean ist KV_BITS=8
#   Default — dort ist das der NORMALBETRIEB, keine Vorsorge.
#   ZWEI PRs FUER DIESELBE SACHE: #1956 (hier) und #1938 ("Fix Qwen speculative
#   decoding with quantized batch cache") aendern dieselben zwei Dateien mit
#   demselben Inhalt. Nur einer wird mergen — dieser Patch deckt beide ab.
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
# 0020-dflash2-qwen38.patch   ERSETZT 2026-08-20 durch 0040 (Upstream-PR #1959).
#   Die eigene Transkription aus z-lab/dflash war korrekt — inklusive des
#   Codebook-Renames, den z-lab erst am 2026-08-18 mit e128a7e kanonisierte und
#   den #1959 identisch macht. Ersetzt wurde sie trotzdem: #1959 bringt den
#   exakten 4bit-M=4-Verifier-Kernel und verteilungserhaltendes Rejection
#   Sampling dazu. Liegt in der Git-Historie dieses Repos.
#
# 0022-dflash-guard-invalid-bonus-token.patch   ERSETZT 2026-08-20 durch 0041.
#   Gleicher Guard, andere Stelle: DFlash 2 baut den Block seit #1959 in
#   DFlash2DraftModel.propose_block statt in DFlashDraftModel.draft_block.
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

# Zuruecknehmen geht in UMGEKEHRTER Reihenfolge. Seit 0041 auf Code sitzt, den
# 0040 erst anlegt, ist die Kette geordnet: wer 0040 vor 0041 zurueckzunehmen
# versucht, findet den erwarteten Kontext nicht mehr, der Reverse-Dry-Run
# schlaegt fehl und der Patch bliebe still drin. (${(Oa)...} kehrt um.)
# WICHTIG: revert_order bleibt eine eigene Variable. Wuerde man `patches` selbst
# umdrehen, drehte die Probe unten es ein zweites Mal zurueck — dann prueft sie
# in Anwende- statt Abtragereihenfolge und meldet 0040 als "nicht angewendet".
revert_order=( ${(Oa)patches} )

# ── Angewendet-Status ermitteln ──────────────────────────────────────────────
# Der Reverse-Dry-Run allein reicht nicht mehr, seit Patches aufeinander sitzen:
# 0041 liegt MITTEN in dem Code, den 0040 erst anlegt. Ein Reverse-Dry-Run von
# 0040 findet dann seinen eigenen Kontext nicht wieder und meldet ihn faelschlich
# als offen — worauf ein zweiter Lauf ihn erneut anwenden wollte und "KONFLIKT"
# schrie.
# Deshalb wird der Status auf einer KOPIE ermittelt, die ruecklaeufig abgetragen
# wird: erst 0041 zurueck, dann steht 0040 wieder frei da. 15 MB, einmal pro Lauf.
_probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlxvlm-probe.XXXXXX")
trap 'rm -rf "$_probe_dir"' EXIT INT TERM
cp -R "$SITE_PACKAGES/mlx_vlm" "$_probe_dir/"
typeset -A applied_map
for p in "${revert_order[@]}"; do
  if patch -R -p1 --dry-run --force -d "$_probe_dir" < "$p" &>/dev/null; then
    applied_map[${p:t}]=1
    patch -R -p1 --force -d "$_probe_dir" < "$p" &>/dev/null || true
  else
    applied_map[${p:t}]=0
  fi
done
rm -rf "$_probe_dir"
trap - EXIT INT TERM

[[ "$MODE" == "revert" ]] && patches=( "${revert_order[@]}" )

for p in "${patches[@]}"; do
  name="${p:t}"
  applied=${applied_map[$name]}

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
