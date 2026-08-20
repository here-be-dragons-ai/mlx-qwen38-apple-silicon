#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm Server Start-Skript  –  Qwen3.8 27B (DENSE, MLX 4bit)
# ZIELHARDWARE:  Apple Silicon ab 32 GB Unified Memory
#
# Die speicherrelevanten Parameter kommen aus PROFILE (s.u.):
#     lean      32 GB ohne sudo   (Working-Set 21,3 GiB)   Peak ~18,8 GiB
#     balanced  32 GB mit sysctl  (Working-Set 26 GiB)     Peak ~25,0 GiB
#     roomy     48 GB             (Working-Set 44 GiB)     Peak ~41 GiB
#     auto      (Default) waehlt anhand des Working-Sets
#
# Die MESSWERTE in diesem Skript stammen alle von der 48-GB-Maschine (M5 Pro,
# mlx-vlm 0.6.13/0.6.15), auf der das "roomy"-Profil ueber Wochen lief. Fuer 32 GB
# uebernommen sind die hardwareunabhaengigen Erkenntnisse (MTP-SpecDec lohnt,
# APC lohnt, KV-Fensterung halluziniert); die SPEICHER-Parameter sind gerechnet.
#
# ── PASST DAS MODELL AUF 32 GB? JA — aber der Kontext ist das Nadeloehr. ─────
#
#   Gewichte        14,95 GiB (3 Shards, 4bit affine, group_size 64)
#   MTP-Drafter      0,23 GiB (mlx-community/Qwen3.8-27B-MTP-4bit)
#   ──────────────────────────
#   fix belegt      ~15,2 GiB   <- passt auch in den macOS-Default (s.u.)
#
#   Dazu kommt PRO SEQUENZ UND PRO APC-SNAPSHOT:
#     KV-Cache      64 KiB/Token  (16 Full-Attn-Layer x 4 KV-Heads x 512 x 2 B)
#                                  mit KV_BITS=8: 32 KiB, mit KV_BITS=4: 16 KiB
#     GDN-State    ~152 MiB fix   (48 Linear-Layer x 48 v-Heads x 128 x 128 x 4 B,
#                                  mamba_ssm_dtype=float32; laengenunabhaengig)
#
#   Metals max_recommended_working_set_size ist auf Macs <= 36 GB per Default
#   2/3 des RAM  →  32 GB = 21,33 GiB. Davon bleiben nach den Gewichten nur
#   ~4,6 GiB fuer KV + Snapshots + Aktivierungen. Deshalb:
#
#     OHNE sysctl (21,3 GiB Budget) :  ~23k Token Kontext  → Hermes ctx 24576
#     OHNE sysctl, dafuer KV_BITS=8 :  ~46k Token Kontext  → Hermes ctx 32768
#     MIT  sysctl 26624 MB (26 GiB) :  ~48k Token Kontext  → Hermes ctx 49152
#     MIT  sysctl + KV_BITS=8       :  ~97k Token Kontext  → Hermes ctx 65536
#
#   Das Skript RECHNET dieses Budget beim Start aus den echten Werten der
#   laufenden Maschine aus und druckt es (Abschnitt "Speicherbudget") — die
#   Zahlen oben sind nur die Erwartung fuer eine nackte 32-GB-Maschine.
#
#   Wired-Limit setzen (nicht persistent, braucht sudo):
#       sudo sysctl -w iogpu.wired_limit_mb=26624
#   Persistent: com.local.iogpu-wired-limit.plist aus diesem Verzeichnis, s. README.
#   NICHT hoeher als 26624 auf 32 GB — darunter braucht macOS selbst ~5-6 GB;
#   wer die Grenze zu hoch setzt, tauscht Metal-OOM gegen Kernel-Panic/Beachball.
#
# ── ERWARTETE GESCHWINDIGKEIT (GESCHAETZT, nicht auf M5-Basis gemessen) ──────
#   Das Modell ist DENSE: jeder Decode-Schritt liest ~15 GiB → reine
#   Bandbreitenfrage. M5 Basis hat ~153 GB/s, der gemessene M5 Pro deutlich
#   mehr. Skaliert von den 48-GB-Messwerten (17,5-18,4 t/s Decode):
#       Decode roh            ~8-10 t/s
#       Decode mit MTP-SpecDec ~13-20 t/s bei Tool-Calls/JSON
#                              (Acceptance dort gemessen 90-93 %, bei Prosa 42 %)
#       Prefill               ~180-250 t/s (M5 Pro: 420-470 t/s, GPU-Cores/2)
#   Folge fuer den Agent-Betrieb: ein KALTER 30k-Prefill dauert ~2-3 Minuten.
#   Genau deshalb sind APC + SSD-Tier hier noch wichtiger als auf der 48-GB-Kiste
#   (dort gemessen: 89 630 ms → 350 ms bei 36k Token, Faktor 256).
#
# Voraussetzung: install-prereqs.sh gelaufen, Port 8888 frei.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Pfade (alle per Env ueberschreibbar; kein hartkodierter Benutzername) ─────
VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"
MODELS_ROOT="${MLX_MODELS:-$HOME/src/mlx/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/Qwen3.8-27B-MLX-4bit}"
LOG_FILE="${LOG_FILE:-$HOME/.hermes/logs/mlx-qwen3.8.log}"
# NICHT $HOST benutzen: zsh belegt diesen Parameter selbst mit dem Hostnamen,
# ein "${HOST:-127.0.0.1}" wuerde den Server auf die LAN-Adresse binden.
BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8888}"

# ── Modell-Alias ──────────────────────────────────────────────────────────────
# mlx-vlm hat KEIN --alias: der "model"-String aus dem Request ist der LADEPFAD.
# Weicht er vom Preload-Pfad ab, wirft der Server das geladene Modell weg und
# startet ein snapshot_download() gegen HuggingFace (→ 401, obwohl lokal da).
# Loesung: Symlink mit genau dem Alias-Namen neben das Modell, Server mit dem
# RELATIVEN Namen vorladen (cd auf MODELS_ROOT am Ende des Skripts).
MODEL_ALIAS="${MODEL_ALIAS:-Qwen3.8-27B-local}"

# ── Profile ───────────────────────────────────────────────────────────────────
# Ein Profil setzt nur DEFAULTS — jede einzeln gesetzte Env-Variable gewinnt
# weiterhin, z. B.  PROFILE=lean APC_ENTRIES=2 ./start-mlx_qwen3.8.sh
#
#   lean      Minimaler RAM. Peak ~18,8 GiB bei 29k-Spitzenprompt → passt unter
#             das macOS-Default-Working-Set von 21,33 GiB, also OHNE sudo.
#             Gerechnete Ersparnis gegenueber balanced (43k-Prompt):
#               APC_ENTRIES 2→1 (3→2 Kopien)  -2,8 GiB   billigster Hebel: der
#                  SSD-Tier bleibt, ein verdraengter Snapshot ist in 350 ms
#                  zurueck statt in 90 s Vollprefill
#               KV_BITS=8 (64→32 KiB/Token)   -4,2 GiB   Durchsatzkosten auf dem
#                  MLX-Pfad NICHT gemessen — bei llama.cpp kostete KV-Quant an
#                  vergleichbarer Stelle bis 8x Prefill. Vor Dauerbetrieb A/B.
#               PREFILL_STEP 1024→512         ~-0,2 GiB  transienter Peak
#
#   balanced  32 GB mit angehobenem Wired-Limit (26624 MB). Peak ~25 GiB bei
#             43k-Spitzenprompt, KV unquantisiert — kein ungemessener Trade-off.
#             (Alias: "default", historischer Name.)
#
#   roomy     48 GB. Das urspruengliche, ueber Wochen gefahrene M5-Pro-Setup:
#             APC_ENTRIES 4 (mehr Konversationen warm), PREFILL_STEP 2048,
#             VISION_CACHE 20, APC_DISK_MAX_GB 60. Peak ~41 GiB bei 110k-Prompt
#             → braucht iogpu.wired_limit_mb=45056.
#             ACHTUNG, die APC-Rechnung unterschaetzt den Agent-Fall: ein
#             Snapshot bei 104k Token ist 6,5 GiB, VIER davon 26 GiB. Deshalb
#             APC_ENTRIES hier NICHT weiter anheben.
#
# Zu jedem Profil gehoert clientseitig ein passendes context_length — der Banner
# druckt es, und weiter unten warnt das Skript, wenn die config.yaml darueber
# liegt.
PROFILE="${PROFILE:-auto}"

# auto: waehlt anhand des Metal-Working-Sets. Der wird hier ueber sysctl
# geschaetzt statt ueber mx.device_info(), weil die Profil-Defaults gebraucht
# werden, BEVOR das venv-Python laeuft (Modell-Load dauert). Regel:
# iogpu.wired_limit_mb gewinnt, wenn gesetzt; sonst der macOS-Default (2/3 des
# RAM bei <= 36 GB, sonst 3/4). Die exakte Zahl aus Metal steht spaeter im
# Banner — weichen beide ab, gilt die aus dem Banner.
if [[ "$PROFILE" == "auto" ]]; then
  _RAM_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))
  _WIRED_MB=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
  if [[ "${_WIRED_MB:-0}" -gt 0 ]]; then
    _WS_MB=$_WIRED_MB
  elif [[ "$_RAM_MB" -le 36864 ]]; then
    _WS_MB=$(( _RAM_MB * 2 / 3 ))
  else
    _WS_MB=$(( _RAM_MB * 3 / 4 ))
  fi
  if   [[ "$_WS_MB" -ge 30720 ]]; then PROFILE=roomy      # >= 30 GiB
  elif [[ "$_WS_MB" -ge 24576 ]]; then PROFILE=balanced   # >= 24 GiB
  else                                 PROFILE=lean
  fi
  _AUTO_NOTE=" (auto, Working-Set ~$(( _WS_MB / 1024 )) GiB)"
else
  _AUTO_NOTE=""
fi

case "$PROFILE" in
  # VISION_CACHE hier 1 und NICHT 0: VisionFeatureCache.put() prueft
  # `len(cache) >= max_size` und ruft dann popitem() — bei max_size=0 also auf
  # ein leeres OrderedDict, was den ersten Bild-Request mit KeyError killt.
  # 1 ist das echte Minimum (vision_cache.py:60ff).
  lean)
    _APC_ENTRIES=1; _KV_BITS="8"; _PREFILL=512;  _VISION=1;  _APC_MAXGB=40; _APC_MINFREE=4.0; _CTX_HINT=32768 ;;
  balanced|default)
    PROFILE=balanced
    _APC_ENTRIES=2; _KV_BITS="";  _PREFILL=1024; _VISION=4;  _APC_MAXGB=40; _APC_MINFREE=4.0; _CTX_HINT=49152 ;;
  # _CTX_HINT hier 98304 und NICHT 131072, obwohl das 48-GB-Setup produktiv mit
  # 131072 lief: die Budgetrechnung unten ist ein WORST CASE (alle APC-Snapshots
  # gleichzeitig auf voller Promptlaenge). Bei 44 GiB Working-Set, 5 Kopien und
  # 64 KiB/Token sind das ~87k Token, also ctx <= ~101k. Dass 131072 in der
  # Praxis trug, liegt daran, dass die vier Snapshots real nie alle gleichzeitig
  # am Anschlag stehen — es ist Glueck, keine Garantie. Wer 131072 will, nimmt
  # APC_ENTRIES=2 dazu (3 Kopien → Budget ~145k).
  roomy)
    _APC_ENTRIES=4; _KV_BITS="";  _PREFILL=2048; _VISION=20; _APC_MAXGB=60; _APC_MINFREE=2.0; _CTX_HINT=98304 ;;
  *)
    echo "ERROR: unbekanntes PROFILE='$PROFILE' (gueltig: auto | lean | balanced | roomy)" >&2; exit 1 ;;
esac

# ── Speculative Decoding (MTP) ────────────────────────────────────────────────
# DEFAULT AN. Auf der 48-GB-Maschine am 2026-08-17 durchgemessen:
#   Durchsatz : Decode 16,9-18,3 → 26,9-41,5 t/s (+58..+132 %)
#   Acceptance: 42 % Prosa, 90 % JSON, 93 % Tool-Call
#   Qualitaet : 7/7 Antworten BIT-IDENTISCH zu ohne Drafter (temperature 0)
#   APC       : ueberlebt (Turn 2 cached 3178 von 3217) — das cached_tokens=0-
#               Problem aus mlx-vlm 0.6.12 existiert in 0.6.13 nicht mehr.
# Fuer 32 GB besonders attraktiv: der Drafter kostet nur 0,23 GiB, der Gewinn
# ist reiner Bandbreiten-Gewinn — und Bandbreite ist auf dem M5 Basis knapp.
# Rollback: ENABLE_SPEC_DECODE=0 ./start-mlx_qwen3.8.sh
ENABLE_SPEC_DECODE="${ENABLE_SPEC_DECODE:-1}"
# DRAFT_KIND=mtp|dflash
#   mtp     MTP-Kopf, 0,23 GiB. Default, weil ueber Wochen gemessen:
#           +58..132 % Decode, Acceptance 90 % JSON / 93 % Tool-Call,
#           Ausgabe 7/7 bit-identisch.
#   dflash  DFlash 2 (z-lab), 1,01 GiB in 4bit. Block-Diffusion-Drafter mit
#           Pfad-Selektor. Braucht den lokalen Patch 0020 — mlx-vlm selbst
#           implementiert nur DFlash v1.
#           ACHTUNG BLOCKGROESSE: der Checkpoint ist auf block_size 8 ausgelegt,
#           z-lab empfiehlt fuer quantisierte MLX-Modelle aber <= 5, und die
#           eigene Kernelmessung zeigt bei M=5 eine Dispatch-Klippe
#           (M=1 5,7 ms, M=4 6,3 ms, M=5 7,4 ms). Default hier deshalb 4.
DRAFT_KIND="${DRAFT_KIND:-mtp}"
case "$DRAFT_KIND" in
  mtp)    _DRAFT_DEFAULT="$MODELS_ROOT/Qwen3.8-27B-MTP-4bit";      _BLOCK_DEFAULT="" ;;
  dflash) _DRAFT_DEFAULT="$MODELS_ROOT/Qwen3.8-27B-DFlash2-4bit";  _BLOCK_DEFAULT="4" ;;
  *) echo "ERROR: unbekanntes DRAFT_KIND='$DRAFT_KIND' (gueltig: mtp | dflash)" >&2; exit 1 ;;
esac
DRAFT_MODEL="${DRAFT_MODEL:-$_DRAFT_DEFAULT}"
DRAFT_BLOCK_SIZE="${DRAFT_BLOCK_SIZE-$_BLOCK_DEFAULT}"

# ── Automatic Prefix Caching ──────────────────────────────────────────────────
# Wiederverwendet den KV-Cache, wenn der neue Prompt den alten als Prefix
# enthaelt (= jeder Folge-Turn). Seit mlx-vlm 0.6.13 upstream korrekt.
ENABLE_APC="${ENABLE_APC:-1}"
# 2 STATT 4 (48-GB-Skript) STATT 8 (Qwen3.6): ein Snapshot ist so gross wie der
# Prompt, und dieses Modell braucht 64 KiB/Token. Auf 32 GB ist jeder weitere
# Eintrag direkt weniger Kontext — s. Budgetrechnung im Banner.
# Zusammen mit APC_SINGLE=1 (unten) heisst 2 == zwei Konversationen warm.
APC_ENTRIES="${APC_ENTRIES:-$_APC_ENTRIES}"
# SSD-Tier: ueberlebt Serverneustarts, gemessen Faktor 256 auf einen kalten
# 36k-Prefill. Auf dieser Maschine noch wertvoller, weil der Prefill langsamer
# ist. Preis: ~2,3 GB Schreiblast pro 36k-Konversation → Deckel unten.
APC_DISK="${APC_DISK:-$HOME/.hermes/apc}"
APC_DISK_MAX_GB="${APC_DISK_MAX_GB:-$_APC_MAXGB}"
# Der Disk-Tier restauriert nur, wenn noch so viel RAM frei ist. Upstream-Default
# 2.0 ist fuer 32 GB zu knapp — ein Restore mitten in den Speicherdruck hinein
# ist genau der Weg in "[METAL] Insufficient Memory".
APC_MIN_FREE_RAM_GB="${APC_MIN_FREE_RAM_GB:-$_APC_MINFREE}"
# Unterdrueckt den redundanten Voll-Snapshot (getroffen wird immer der
# Checkpoint bei len-16). Halbiert den APC-Speicher. Braucht den lokalen Patch
# 0010 (patches/apply-patches.sh) — ohne ihn ist die Variable wirkungslos, das
# Skript warnt dann.
APC_SINGLE="${APC_SINGLE:-1}"

# ── Prefill / Slots ───────────────────────────────────────────────────────────
# 1024 STATT 2048 (48-GB-Skript): der Prefill-Chunk bestimmt den transienten
# Aktivierungs-Peak. Auf der 48-GB-Maschine wurde gemessen, dass der Prefill
# rechen- und nicht chunklimitiert ist (2048 vs 4096: 94,6 s vs 94,2 s auf
# denselben Prompt) — die Halbierung kostet also fast nichts und kauft
# Kopffreiheit. Bei viel Luft im Budget: PREFILL_STEP=2048 ./start-...
PREFILL_STEP="${PREFILL_STEP:-$_PREFILL}"
# 1 Slot. Auf einem dichten Modell wuerde Batch 2 im Aggregat fast linear
# skalieren (gemeinsamer Gewichts-Read), aber jede zusaetzliche Sequenz kostet
# einen kompletten KV-Satz + GDN-State — auf 32 GB nicht drin.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
# 20 → 4: Vision-Features sind gecachte Bild-Embeddings, hier reine Speicherlast.
VISION_CACHE="${VISION_CACHE:-$_VISION}"
LOG_PROGRESS="${LOG_PROGRESS:-10}"

# ── KV-Quantisierung ──────────────────────────────────────────────────────────
# DEFAULT AUS, aber der wichtigste Hebel dieser Maschine: KV_BITS=8 halbiert
# 64 → 32 KiB/Token und verdoppelt damit den moeglichen Kontext.
#     KV_BITS=8 QUANT_KV_START=8192 ./start-mlx_qwen3.8.sh
# QUANT_KV_START laesst die ersten 8k Token unquantisiert — kurze Turns bleiben
# damit voll schnell, nur lange zahlen die Dequantisierungskosten.
# WARUM NICHT DEFAULT AN: Fuer llama.cpp/Qwen3.6 wurde 2026-08-11 gemessen, dass
# KV-Quantisierung bis zu 8x Prefill und 1,9x Decode KOSTET (Dequant-Kosten
# wachsen mit der KV-Laenge). Fuer den MLX-Pfad ist das NICHT nachgemessen. Auf
# einer Maschine mit ~9 t/s Decode waere ein solcher Faktor fatal — also erst
# messen, dann einschalten:
#     A/B mit gleichem Prompt, Decode-t/s aus dem Log vergleichen.
# Wenn der Kontext ohnehin <= 40k bleibt: aus lassen, nichts gewonnen.
# ${VAR-...} ohne Doppelpunkt: nur wenn KV_BITS UNGESETZT ist, greift das Profil.
# Leer ist hier ein gueltiger Wert (= f16), deshalb muss
#   PROFILE=lean KV_BITS= ./start-mlx_qwen3.8.sh
# die Quantisierung wieder abschalten koennen.
KV_BITS="${KV_BITS-$_KV_BITS}"
KV_SCHEME="${KV_SCHEME:-}"
QUANT_KV_START="${QUANT_KV_START:-8192}"

# ── NICHT BENUTZEN: --max-kv-size ─────────────────────────────────────────────
# mlx-vlm kann den KV-Cache hart deckeln (rotierendes Fenster). Das ist auf 32 GB
# verlockend und trotzdem falsch: die aequivalente Idee (Gleitfenster nur auf den
# 16 Full-Attn-Layern, GDN-State voll) wurde am 2026-08-17 mit needle_hybrid.py
# WIDERLEGT — die Nadel ausserhalb des Fensters ging verloren, im 40-%-Fall
# halluzinierte das Modell sogar eine falsche Zahl (8347 statt 8342). Lieber
# kleiner Kontext + Kompaktierung im Agenten als ein stiller Qualitaetsverlust.

# ── Pruefungen ────────────────────────────────────────────────────────────────
[[ -x "$VENV_PY" ]] || {
  echo "ERROR: venv-Python nicht gefunden: $VENV_PY"
  echo "       Erst einrichten:  ./install-prereqs.sh"
  exit 1
}
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: Modell nicht gefunden: $MODEL_DIR"
  echo "       ./download-mlx-model.sh mlx-community/Qwen3.8-27B-4bit $MODEL_DIR"
  exit 1
fi

# Vollstaendigkeit pruefen: alle in der Index-Datei referenzierten Shards da?
# (Ein abgebrochener 15-GB-Download faellt sonst erst nach dem Modell-Load auf.)
if [[ -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  MISSING=$(python3 -c "
import json,os
d=json.load(open('$MODEL_DIR/model.safetensors.index.json'))
want=sorted(set(d['weight_map'].values()))
print(' '.join(f for f in want if not os.path.exists(os.path.join('$MODEL_DIR',f))))")
  if [[ -n "$MISSING" ]]; then
    echo "ERROR: Modell unvollstaendig — fehlende Shards: $MISSING"
    echo "       ./download-mlx-model.sh mlx-community/Qwen3.8-27B-4bit $MODEL_DIR"
    exit 1
  fi
fi

if lsof -iTCP:$PORT -sTCP:LISTEN -n &>/dev/null; then
  echo "WARNUNG: Port $PORT ist bereits belegt."
  echo "  → mlx-vlm beenden:     pkill -f mlx_vlm.server"
  echo "  → llama-server beenden: pkill -f llama-server"
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

# ── Patch-Checks ──────────────────────────────────────────────────────────────
SITE_PACKAGES=$("$VENV_PY" -c "import mlx_vlm,os;print(os.path.dirname(os.path.dirname(mlx_vlm.__file__)))")
MLX_VLM_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx-vlm'))" 2>/dev/null || echo "?")

# APC-Faehigkeit: ab 0.6.13 gibt es semantic_extra_hash(). Fehlt sie (Downgrade),
# trifft der Prefix-Cache nur bei byte-identischen Prompts — kostet aber trotzdem
# Speicher, und Speicher ist hier die knappe Ressource.
if [[ "$ENABLE_APC" == "1" ]] && ! grep -q "def semantic_extra_hash" "$SITE_PACKAGES/mlx_vlm/apc.py" 2>/dev/null; then
  echo "⚠️  WARNUNG: mlx-vlm $MLX_VLM_VER kennt semantic_extra_hash() nicht (< 0.6.13?)." >&2
  echo "    Prefix-Caching greift dann NUR bei byte-identischen Prompts." >&2
  echo "    Fix:  uv pip install -U mlx-vlm" >&2
fi
# Patch 0010 (Einzel-Snapshot). Ohne ihn ist APC_SINGLE wirkungslos und der
# Speicherbedarf pro Konversation doppelt so hoch wie unten gerechnet.
APC_SINGLE_OK=0
grep -q "QWEN38_APC_SINGLE_SNAPSHOT" "$SITE_PACKAGES/mlx_vlm/generate/ar.py" 2>/dev/null && APC_SINGLE_OK=1
if [[ "$APC_SINGLE" != "0" && "$APC_SINGLE_OK" == "0" ]]; then
  echo "⚠️  WARNUNG: Patch 0010 fehlt — APC_SINGLE ist wirkungslos, jeder Request" >&2
  echo "    legt ZWEI Snapshots ab (doppelter APC-Speicher, halber Kontext)." >&2
  echo "    Fix:  ./patches/apply-patches.sh" >&2
fi

# ── Drafter pruefen ───────────────────────────────────────────────────────────
SPEC_STATUS="OFF"
if [[ "$ENABLE_SPEC_DECODE" != "0" ]]; then
  if [[ -f "$DRAFT_MODEL/config.json" ]]; then
    SPEC_STATUS="ON"
  else
    echo "  WARNUNG: Drafter nicht gefunden ($DRAFT_MODEL) — starte OHNE SpecDec."
    if [[ "$DRAFT_KIND" == "dflash" ]]; then
      echo "           ./download-mlx-model.sh z-lab/Qwen3.8-27B-DFlash2 ${MODELS_ROOT}/Qwen3.8-27B-DFlash2-bf16"
      echo "           ./convert-dflash2-drafter.py ${MODELS_ROOT}/Qwen3.8-27B-DFlash2-bf16 $DRAFT_MODEL"
    else
      echo "           ./download-mlx-model.sh mlx-community/Qwen3.8-27B-MTP-4bit $DRAFT_MODEL"
    fi
    ENABLE_SPEC_DECODE=0
  fi
fi

# ── Alias-Symlink setzen ──────────────────────────────────────────────────────
ALIAS_LINK="$MODELS_ROOT/$MODEL_ALIAS"
if [[ -e "$ALIAS_LINK" && ! -L "$ALIAS_LINK" ]]; then
  echo "ERROR: $ALIAS_LINK existiert und ist KEIN Symlink — bitte pruefen/entfernen." >&2
  exit 1
fi
ln -sfn "$MODEL_DIR" "$ALIAS_LINK"

# ── Speicherbudget ausrechnen ─────────────────────────────────────────────────
# Rechnet mit den ECHTEN Werten dieser Maschine statt mit den Annahmen im Kopf
# des Skripts: Metal-Working-Set (folgt iogpu.wired_limit_mb), tatsaechliche
# Dateigroessen, gewaehlte KV-Bits, gewaehlte APC-Eintraege.
WEIGHTS_KB=$(du -skL "$MODEL_DIR" | cut -f1)
DRAFT_KB=0
[[ "$ENABLE_SPEC_DECODE" != "0" ]] && DRAFT_KB=$(du -skL "$DRAFT_MODEL" | cut -f1)
# mlx-vlm hat kein -c-Flag: der native Kontext steht in der config.json.
MODEL_CTX=$(python3 -c "
import json
d=json.load(open('$MODEL_DIR/config.json'))
tc=d.get('text_config',d)
print(tc.get('max_position_embeddings') or d.get('max_position_embeddings') or '')" 2>/dev/null || true)
BUDGET=$(
  WEIGHTS_KB="$WEIGHTS_KB" DRAFT_KB="$DRAFT_KB" KV_BITS="$KV_BITS" MODEL_CTX="$MODEL_CTX" \
  APC_ENTRIES="$APC_ENTRIES" ENABLE_APC="$ENABLE_APC" APC_SINGLE="$APC_SINGLE" \
  APC_SINGLE_OK="$APC_SINGLE_OK" \
  "$VENV_PY" - <<'PY'
import os
import mlx.core as mx

GiB = 1 << 30
info = mx.device_info()
ws = info["max_recommended_working_set_size"]
ram = info["memory_size"]

weights = (int(os.environ["WEIGHTS_KB"]) + int(os.environ["DRAFT_KB"])) * 1024
# Aktivierungen, Metal-Heap-Fragmentierung, Tokenizer, Python. Erfahrungswert von
# der 48-GB-Maschine (RSS-Leerlauf 15,5 GiB bei 15,2 GiB Gewichten, Peak waechst
# mit dem Prompt) — bewusst grosszuegig, weil Unterschaetzen hier OOM heisst.
reserve = int(1.5 * GiB)

kv_bits = os.environ.get("KV_BITS") or ""
per_tok = 16 * 4 * (256 + 256) * 2                      # 65536 B, f16
if kv_bits:
    per_tok = int(per_tok * float(kv_bits) / 16.0)
# GDN/Mamba-State: 48 Linear-Layer x 48 v-Heads x 128 (k) x 128 (v) x 4 B (float32)
recurrent = 48 * 48 * 128 * 128 * 4

copies = 1                                               # die lebende Sequenz
if os.environ["ENABLE_APC"] == "1":
    # APC_EXACT_CACHE_ENTRIES deckelt die Zahl der Snapshots, nicht die Bytes —
    # der Speicherbedarf ist also entries * Promptlaenge, unabhaengig von Patch
    # 0010. Der Patch aendert nur, WIE VIELE KONVERSATIONEN in diese Eintraege
    # passen (mit: eine pro Eintrag, ohne: eine pro zwei Eintraegen).
    copies += int(os.environ["APC_ENTRIES"])

avail = ws - weights - reserve - copies * recurrent
tokens = int(avail / (copies * per_tok)) if avail > 0 else 0
# Nach oben deckelt die Architektur: mehr als max_position_embeddings kann das
# Modell nicht, egal wieviel RAM frei ist.
tokens = min(tokens, int(os.environ.get("MODEL_CTX") or tokens))

print(f"{ws/GiB:.1f}|{ram/GiB:.0f}|{weights/GiB:.1f}|{avail/GiB:.1f}|{tokens}|{copies}|{per_tok//1024}|{info['device_name']}")
PY
)
WS_GIB="${BUDGET%%|*}"; REST="${BUDGET#*|}"
RAM_GIB="${REST%%|*}"; REST="${REST#*|}"
W_GIB="${REST%%|*}";   REST="${REST#*|}"
AVAIL_GIB="${REST%%|*}"; REST="${REST#*|}"
MAX_TOKENS_FIT="${REST%%|*}"; REST="${REST#*|}"
COPIES="${REST%%|*}";  REST="${REST#*|}"
KV_KIB="${REST%%|*}";  DEV_NAME="${REST#*|}"

echo "──────────────────────────────────────────────────────────────"
echo "  mlx-vlm $MLX_VLM_VER  |  Qwen3.8 27B (DENSE, MLX 4bit)  |  $DEV_NAME"
echo "  Profil   :  $PROFILE$_AUTO_NOTE  —  empf. context_length $_CTX_HINT"
echo "  Modell   :  $MODEL_DIR"
echo "  API-Name :  $MODEL_ALIAS  (Symlink; MUSS zum Request-Modellnamen passen)"
echo "  Port     :  $BIND_HOST:$PORT"
echo "  SpecDec  :  $SPEC_STATUS  (Drafter: $DRAFT_KIND, $(du -shL "$DRAFT_MODEL" 2>/dev/null | cut -f1)${DRAFT_BLOCK_SIZE:+, block_size $DRAFT_BLOCK_SIZE})"
if [[ "$ENABLE_APC" == "1" ]]; then
  echo "  APC      :  ON ($APC_ENTRIES Snapshots, Single=$([[ "$APC_SINGLE" != "0" && "$APC_SINGLE_OK" == "1" ]] && echo ja || echo NEIN)${APC_DISK:+, SSD: $APC_DISK})"
else
  echo "  APC      :  OFF"
fi
echo "  KV-Cache :  ${KV_BITS:-f16 unquantisiert} → ${KV_KIB} KiB/Token"
echo "  Prefill  :  Chunk $PREFILL_STEP   Slots: $MAX_NUM_SEQS   Vision-Cache: $VISION_CACHE"
echo "  Log      :  $LOG_FILE"
echo "  ──────────── Speicherbudget (gerechnet, keine Messung) ──────"
echo "  RAM              : ${RAM_GIB} GiB"
echo "  Metal-Working-Set: ${WS_GIB} GiB   (folgt iogpu.wired_limit_mb)"
echo "  Gewichte         : ${W_GIB} GiB   (+1,5 GiB Reserve fuer Aktivierungen)"
echo "  frei fuer KV     : ${AVAIL_GIB} GiB  auf ${COPIES} Kopien (1 live + APC)"
echo "  →  KONTEXT-BUDGET: ~${MAX_TOKENS_FIT} Token pro Konversation"
echo "──────────────────────────────────────────────────────────────"

# Harte Warnung, wenn das Working-Set auf dem 32-GB-Default steht.
if (( $(printf '%.0f' "$WS_GIB") < 24 )) && (( $(printf '%.0f' "$RAM_GIB") >= 30 )); then
  echo "ℹ️  Working-Set ist ${WS_GIB} GiB — das ist der macOS-Default (2/3 RAM)." >&2
  echo "    Mit  sudo sysctl -w iogpu.wired_limit_mb=26624  werden daraus 26 GiB" >&2
  echo "    und aus ~${MAX_TOKENS_FIT} Token rund das Doppelte. Persistent: siehe README." >&2
fi
if [[ -z "$KV_BITS" && "$MAX_TOKENS_FIT" -lt 40000 ]]; then
  echo "ℹ️  Unter 40k Token Budget. Verdoppeln ohne sudo:" >&2
  echo "      KV_BITS=8 QUANT_KV_START=8192 ./start-mlx_qwen3.8.sh   (vorher A/B messen)" >&2
fi

# ── Hermes-Abgleich (rein lesend) ─────────────────────────────────────────────
# mlx-vlm hat KEIN -c-Flag; der Kontext kommt aus der config.json (262144). Der
# effektive Deckel ist also allein Hermes' model.context_length. Steht der ueber
# dem Budget oben, laeuft der Server irgendwann in "[METAL] Insufficient Memory".
HERMES_CFG="$HOME/.hermes/config.yaml"
if [[ -f "$HERMES_CFG" ]]; then
  CONFIG_CTX=$(awk '
    /^model:/ { in_model=1; next }
    in_model && /^[a-zA-Z_]/ { in_model=0 }
    in_model && /context_length:/ { gsub(/[^0-9]/,"",$0); print; exit }
  ' "$HERMES_CFG")
  if [[ -n "$CONFIG_CTX" && "$CONFIG_CTX" -gt "$MAX_TOKENS_FIT" ]]; then
    echo "⚠️  WARNUNG: config.yaml model.context_length=$CONFIG_CTX > Budget $MAX_TOKENS_FIT." >&2
    echo "    Hermes wird Prompts bauen, die hier nicht mehr in den Speicher passen." >&2
    echo "    Entweder context_length senken oder KV_BITS=8 / wired_limit anheben." >&2
  fi
  CONFIG_MODEL=$(awk '
    /^model:/ { in_model=1; next }
    in_model && /^[a-zA-Z_]/ { in_model=0 }
    in_model && /default:/ { sub(/^[^:]*:[[:space:]]*/,""); gsub(/["\x27]/,""); print; exit }
  ' "$HERMES_CFG")
  if [[ -n "$CONFIG_MODEL" && "$CONFIG_MODEL" != "$MODEL_ALIAS" ]]; then
    echo "⚠️  WARNUNG: config.yaml model.default='$CONFIG_MODEL' != Alias '$MODEL_ALIAS'." >&2
    echo "    Bei mlx-vlm IST der Request-Modellname der Ladepfad — Abweichung fuehrt" >&2
    echo "    zu Reload + HF-Download (401). Angleichen oder:" >&2
    echo "      MODEL_ALIAS='$CONFIG_MODEL' ./start-mlx_qwen3.8.sh" >&2
  fi
fi

# ── Thinking ──────────────────────────────────────────────────────────────────
# Serverseitig AUS (mlx-vlm schickt enable_thinking immer ans Template, Default
# false). Hermes/pi.dev setzen pro Request enable_thinking:true +
# reasoning_effort:low. GUELTIG SIND NUR low|medium|xhigh — jeder andere Wert,
# auch "none", wirft im Template eine Exception → HTTP 500. Ohne Angabe
# defaultet das Template auf 'xhigh' (gemessen 1269 statt 428 Completion-Tokens).
# Auf dieser Maschine ist 'low' nicht nur billiger, sondern ueberlebenswichtig:
# bei ~9 t/s sind 1269 Tokens ueber zwei Minuten reines Nachdenken.

args=(
  -m mlx_vlm.server
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

# CWD = models-Root, damit der relative Alias-Name aufgeloest wird
# (get_model_path() macht Path(name).exists() gegen das Arbeitsverzeichnis).
cd "$MODELS_ROOT"

exec "$VENV_PY" "${args[@]}" > >(tee -a "$LOG_FILE") 2>&1
