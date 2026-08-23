#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm Server Start-Skript  –  Qwen3.8 27B (DENSE, MLX 4bit)
# ZIELHARDWARE:  Apple Silicon ab 32 GB Unified Memory
#
# Die speicherrelevanten Parameter kommen aus PROFILE (s.u.):
#     lean      32 GB ohne sudo   (Working-Set 21,3 GiB)   Peak ~18,8 GiB
#     balanced  32 GB mit sysctl  (Working-Set 26 GiB)     Peak ~25,0 GiB
#     roomy     48 GB             (Working-Set 44 GiB)     Peak ~30 GiB (ger.)
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
#     OHNE sysctl (21,3 GiB Budget) :  ~23k Token Kontext  → Client-ctx 24576
#     OHNE sysctl, dafuer KV_BITS=8 :  ~46k Token Kontext  → Client-ctx 32768
#     MIT  sysctl 26624 MB (26 GiB) :  ~48k Token Kontext  → Client-ctx 49152
#     MIT  sysctl + KV_BITS=8       :  ~97k Token Kontext  → Client-ctx 65536
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
# Laufzeitdaten: Log und SSD-Prefix-Cache. Wer sie woanders haben will, setzt
# STATE_DIR — oder LOG_FILE / APC_DISK einzeln.
STATE_DIR="${STATE_DIR:-$HOME/.mlx-qwen38}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"
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
#             PREFILL_STEP 2048, VISION_CACHE 20, APC_DISK_MAX_GB 60.
#             → braucht iogpu.wired_limit_mb=40960. NICHT 45056: das sind
#               44 GiB auf 48 GiB, macOS bleiben ~2 GiB, und genau daran ist
#               die Maschine am 2026-08-21 in eine Kernel-Panik gelaufen.
#             APC_ENTRIES war hier bis 2026-08-20 auf 4 und ist jetzt 2: ein
#             Snapshot ist so gross wie der Prompt (64 KiB/Token), bei 104k
#             Token also 6,5 GiB — VIER davon sind 26 GiB und sprengen das
#             Budget, sobald der Kontext wirklich ausgereizt wird. Mit 2 statt
#             4 Eintraegen (3 statt 5 Kopien) steigt das Kontext-Budget bei
#             44 GiB Working-Set von ~85k auf ~142k Token, der gerechnete Peak
#             bei einem 90k-Prompt faellt von ~41 auf ~30 GiB. Preis: zwei
#             statt vier Konversationen warm — der SSD-Tier holt eine
#             verdraengte in ~350 ms zurueck statt in ~90 s Vollprefill.
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
# Der Working-Set wird IMMER berechnet, nicht nur bei PROFILE=auto: roomy
# entscheidet unten anhand von _WIRED_MB, wieviele APC-Snapshots ins Budget
# passen.
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
  # _CTX_HINT bleibt 98304, obwohl das Budget mit APC_ENTRIES=2 rechnerisch fuer
  # 131072 reicht (3 Kopien, 44 GiB Working-Set → ~142k Token): die 142k galten
  # nur MIT gesetztem wired_limit, und zwar mit den historischen 45056. Steht
  # das Limit auf dem macOS-Default (48 GB → 37,4 GiB Working-Set), sind es
  # ~107k, und 131072 waere wieder ueberbucht. 98304 traegt in beiden Faellen.
  # Alle diese Zahlen sind UNFUSED gerechnet — mit aktivem Patch 0013 faellt der
  # Prefill-Transient weg und die Budgets liegen deutlich hoeher (auf 37,4 GiB
  # meldet das Banner ~161k). Massgeblich ist die Banner-Zeile, nicht dieser
  # Kommentar. Wer 131072 fahren will,
  # setzt erst das wired_limit UND APC_ENTRIES=2 (mit dem neuen Default 3 sind
  # es nur ~109k) und prueft die Budgetzeile im Banner.
  # APC_ENTRIES 3 STATT 2, ABER NUR MIT GESETZTEM WIRED-LIMIT.
  # Gemessen am 2026-08-20 aus dem Produktivlog: 168 von 549 Prefills liefen
  # kalt (30,6 %), 45 davon ueber 8k Token = 2415 s reine Prefill-Zeit. Die
  # teuren Faelle sind KEINE neuen Konversationen — im Fenster 10:03-10:22 lief
  # kein Serverneustart, trotzdem fielen drei Requests auf cached_tokens=0
  # (50,4 s / 56,8 s / 70,7 s) zwischen lauter warmen Turns derselben Groesse.
  # Das ist Verdraengung: mehr als zwei gleichzeitig aktive Konversationen auf
  # zwei Snapshot-Plaetzen. Ein dritter Platz kostet eine KV-Kopie.
  # Die Rechnung entscheidet, ob er passt (64 KiB/Token, +152 MiB GDN-State):
  #   ohne wired_limit (48 GB → 36 GiB WS): 4 Kopien →  77k Token < 98304  NEIN
  #   mit  wired_limit 40960 (40 GiB WS)  : 4 Kopien → ~97k Token           JA
  # Deshalb haengt der Default am tatsaechlich gesetzten Limit statt an einer
  # Annahme. Ohne Limit bleibt es bei 2 — sonst tauscht man kalte Prefills
  # gegen "[METAL] Insufficient Memory", was der schlechtere Handel ist.
  # Die Schwelle stand bis 2026-08-21 auf 45056. Das sind 44 GiB auf einer
  # 48-GiB-Maschine, macOS bleiben ~2 GiB — an dem Wert ist die Maschine an
  # jenem Tag in eine Kernel-Panik gelaufen (watchdog timeout, s. README).
  # 40960 laesst macOS 8 GiB. Falls die Rechnung doch nicht aufgeht, kappt die
  # Ueberbuchungs-Sperre weiter unten die Snapshot-Zahl.
  #
  # PREFILL_STEP haengt ebenfalls am Wired-Limit, und zwar aus einem anderen
  # Grund als APC_ENTRIES: head_dim ist 256, und mlx' Default-Dispatch laesst
  # fused Full-Attention (bis 0.32.1) nur fuer 64/80/128 zu. Die 16
  # Full-Attn-Layer materialisieren deshalb je einen Score-Tensor
  # n_heads x qL x kL x 2 B — bei Chunk 2048 und 38k Kontext sind das 2,8 GiB
  # PRO LAYER, und durch die verzoegerte Auswertung leben mehrere gleichzeitig.
  # GEMESSEN auf 37,4 GiB Working-Set, kalter Prefill, Tier vorher geleert:
  #   Chunk 2048               bis ~30k, danach [METAL] Insufficient Memory
  #   Chunk 2048 + KV_BITS=8   bis ~30k  — KV halbieren bringt NICHTS, die
  #                                        Grenze ist der Score-Tensor
  #   Chunk  512               bis ~38k  ✓
  # Ohne Limit also 512. Mit Limit bleibt 2048 — dort ist Luft, und groessere
  # Chunks sind beim Prefill etwas effizienter.
  # DAS GILT NUR UNFUSED. Seit dem mlx-Quellbau (0.32.2.dev, 2026-08-21) ist
  # Patch 0013 scharf, der Score-Tensor faellt weg, und die 512 werden weiter
  # unten auf 1024 angehoben — der NAX-Pfad aus PR #3842 verlangt qL >= 1024.
  # Die Zeilen hier bleiben stehen, weil sie ohne Quellbau wieder gelten.
  # ── APC_ENTRIES haengt NICHT mehr am Wired-Limit ────────────────────────────
  # Bis 2026-08-21 stand hier "3 Snapshots, sobald das Limit gesetzt ist",
  # begruendet ueber die Budgetrechnung weiter unten (1 + APC_ENTRIES Kopien
  # a 64 KiB/Token). Diese Rechnung ist WIDERLEGT. Gemessen am 2026-08-21 mit
  # dem Speicher-Sampler (Working-Set 40 GiB, APC_ENTRIES=3, KV_BITS=8):
  #   nach Modell+Drafter, idle   active = 15,96 GiB   <- Baseline, sauber
  #   nach 1 Request a 13.112 Tok active = 27,35 GiB   <- +11,4 GiB
  #   nach 4 Requests             active = 35,54 GiB
  #   bei  23.091 Tok             active = 37,78 GiB   -> 95 %, danach OOM
  # Das Modell erwartet fuer diesen einen 13k-Request 1,6 GiB. Real sind es
  # 11,4 — Faktor 7. Und active faellt zwischen den Requests NICHT zurueck.
  # Solange nicht geklaert ist, wohin die Differenz geht, ist jede Erhoehung
  # der Snapshot-Zahl eine Wette gegen eine Rechnung, die nachweislich luegt.
  # Deshalb konservativ 1. Wer mehr will, setzt APC_ENTRIES explizit und liest
  # die mem-Zeilen im Log mit.
  #
  # Nebenbefund derselben Messung: KV_BITS=8 greift den Verbraucher NICHT an.
  # apc_adapters.py:515 ruft beim Snapshot-Store dequantize_for_apc() — der
  # lebende Cache schrumpft auf 32 KiB/Token, die Snapshots bleiben f16 bei 64.
  # Gekostet hat es 22,9 -> 18,7 tok/s Decode (Mittel aus 6 bzw. 8 Requests).
  # Deshalb bleibt _KV_BITS hier leer.
  roomy)
    _APC_ENTRIES=1
    if [[ "${_WIRED_MB:-0}" -ge 40960 ]]; then _PREFILL=2048; else _PREFILL=512; fi
    _KV_BITS="";  _VISION=20; _APC_MAXGB=80; _APC_MINFREE=2.0; _CTX_HINT=98304 ;;
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
#   mtp     MTP-Kopf, 0,23 GiB. Frueherer Default, weiter gepflegt:
#           +58..132 % Decode, Acceptance 90 % JSON / 93 % Tool-Call,
#           Ausgabe 7/7 bit-identisch.
#   dflash  DEFAULT seit 2026-08-20. DFlash 2 (z-lab), 1,01 GiB in 4bit.
#           Gemessen gegen MTP bei identischen Prompts (Decode aus predicted_ms):
#             JSON  37,8 -> 44,2 t/s   Code 35,8 -> 42,6   Langkontext 36,0 -> 40,8
#             Prosa 29,8 -> 29,9 t/s   (dort liegt MTP bei der Acceptance vorn)
#           Der Gewinn steckt also in STRUKTURIERTER Ausgabe — Tool-Calls, JSON,
#           Code — und damit genau in der Agentenlast. Die Acceptance-RATE ist
#           bei beiden gleich (Median 80 % vs 81 %); DFlash 2 draftet pro Runde
#           mehr Token (block_size 4 statt 3) und gewinnt darueber.
#           Block-Diffusion-Drafter mit
#           Pfad-Selektor. Braucht den lokalen Patch 0020 — mlx-vlm selbst
#           implementiert nur DFlash v1.
#           ACHTUNG BLOCKGROESSE: der Checkpoint ist auf block_size 8 ausgelegt,
#           z-lab empfiehlt fuer quantisierte MLX-Modelle aber <= 5, und die
#           eigene Kernelmessung zeigt bei M=5 eine Dispatch-Klippe
#           (M=1 5,7 ms, M=4 6,3 ms, M=5 7,4 ms). Default hier deshalb 4.
DRAFT_KIND="${DRAFT_KIND:-dflash}"
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
# 2 IN ALLEN PROFILEN (Qwen3.6 stand auf 8): ein Snapshot ist so gross wie der
# Prompt, und dieses Modell braucht 64 KiB/Token. Jeder weitere Eintrag ist
# direkt weniger Kontext — s. Budgetrechnung im Banner. Auch auf 48 GB kostet
# der Sprung von 2 auf 4 rund ein Drittel des Budgets (142k → 85k Token), und
# der Verlust bei einem Cache-Miss ist dank SSD-Tier klein (~350 ms Restore).
# Zusammen mit APC_SINGLE=1 (unten) heisst 2 == zwei Konversationen warm.
APC_ENTRIES="${APC_ENTRIES:-$_APC_ENTRIES}"
# SSD-Tier: ueberlebt Serverneustarts, gemessen Faktor 256 auf einen kalten
# 36k-Prefill. Auf dieser Maschine noch wertvoller, weil der Prefill langsamer
# ist. Preis: ~2,3 GB Schreiblast pro 36k-Konversation → Deckel unten.
APC_DISK="${APC_DISK:-$STATE_DIR/apc}"
APC_DISK_MAX_GB="${APC_DISK_MAX_GB:-$_APC_MAXGB}"
# Der Deckel ist nur eine Obergrenze — er muss auch auf die Platte passen.
# Gemessen am 2026-08-20: der Tier stand mit 58 GB exakt am 60-GB-Deckel und
# raeumte bei praktisch jedem Store ("APC disk: evicted 6 shard(s); now 56341.8
# MB / 64424.5 MB cap"). Damit ist die Rueckfallebene fuer verdraengte
# RAM-Snapshots loechrig, und genau dann kostet ein Miss die vollen 50-70 s.
# roomy geht deshalb auf 80 GB — aber nur soweit das Volume es traegt.
_APC_RESERVE_GB=25
if [[ -d "$APC_DISK" ]]; then
  _apc_used_gb=$(( $(du -sk "$APC_DISK" 2>/dev/null | cut -f1) / 1048576 ))
else
  _apc_used_gb=0
fi
_apc_free_gb=$(( $(df -k "${APC_DISK:h}" 2>/dev/null | tail -1 | awk '{print $4}') / 1048576 ))
_apc_cap_max=$(( _apc_used_gb + _apc_free_gb - _APC_RESERVE_GB ))
if [[ "$_apc_cap_max" -lt 10 ]]; then _apc_cap_max=10; fi
if [[ "$APC_DISK_MAX_GB" -gt "$_apc_cap_max" ]]; then
  echo "  Hinweis: APC_DISK_MAX_GB ${APC_DISK_MAX_GB} → ${_apc_cap_max} GB gekappt" \
       "(nur ${_apc_free_gb} GB frei, ${_APC_RESERVE_GB} GB Reserve)" >&2
  APC_DISK_MAX_GB=$_apc_cap_max
fi
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
# Merken, ob der Wert vom Aufrufer kam: nur der Profil-Default darf unten
# angehoben werden, wenn der fused Pfad die Chunk-Groesse entkoppelt.
_PREFILL_FROM_ENV=0
[[ -n "${PREFILL_STEP:-}" ]] && _PREFILL_FROM_ENV=1
PREFILL_STEP="${PREFILL_STEP:-$_PREFILL}"
# 1 Slot. Auf einem dichten Modell wuerde Batch 2 im Aggregat fast linear
# skalieren (gemeinsamer Gewichts-Read), aber jede zusaetzliche Sequenz kostet
# einen kompletten KV-Satz + GDN-State — auf 32 GB nicht drin.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
# 20 → 4: Vision-Features sind gecachte Bild-Embeddings, hier reine Speicherlast.
VISION_CACHE="${VISION_CACHE:-$_VISION}"
LOG_PROGRESS="${LOG_PROGRESS:-10}"
# Sekunden zwischen zwei Speicher-Messungen im Log; 0 schaltet den Sampler ab.
# Begruendung und Implementierung stehen unten beim exec.
_MEM_PROBE_INTERVAL="${MEM_PROBE_INTERVAL:-5}"

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
MLX_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx'))" 2>/dev/null || echo "?")

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

# Patch 0013 (fused Full-Attention). head_dim ist 256, und ohne den fused Pfad
# materialisiert JEDER der 16 full_attention-Layer einen Score-Transienten von
# n_heads x qL x kL x 2 B. Bei 24 Heads, Chunk 512 und 23k Kontext sind das
# 533 MiB pro Layer — das ist die echte Prefill-Decke, und sie taucht in der
# KV-Budgetrechnung unten NICHT auf. Der Patch kann nur greifen, wenn mlx
# force_fused ueberhaupt kennt (>= 0.32.2); darunter ist er absichtlich inert.
# Deshalb der Laufzeit-Probe statt eines reinen grep: "Patch liegt im venv"
# und "Pfad ist aktiv" sind hier zwei verschiedene Aussagen.
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
  FUSED_STATUS="unfused (per QWEN38_FORCE_FUSED_SDPA=0 abgeschaltet)"
elif [[ "$_PATCH0013" == "0" ]]; then
  FUSED_STATUS="unfused (Patch 0013 fehlt — ./patches/apply-patches.sh)"
elif [[ "$_FUSED_PROBE" == "ok" ]]; then
  FUSED_STATUS="fused (Patch 0013 aktiv)"
  FUSED_OK=1
else
  FUSED_STATUS="unfused (Patch 0013 inert: mlx $MLX_VER kennt force_fused nicht, braucht >= 0.32.2)"
fi

# Mit fused Full-Attention faellt der Score-Transient weg — genau der Grund, aus
# dem roomy ohne Wired-Limit auf Chunk 512 heruntergeht. Zusaetzlich verlangt der
# NAX-Pfad (PR #3842) ausdruecklich qL >= 1024:
#   query_sequence_length >= 1024 && query_head_dim == 256 && do_causal
# Bei 512 greift er also gar nicht erst. Nur den Profil-Default anheben, eine
# explizite Vorgabe des Aufrufers bleibt unangetastet.
_PREFILL_NOTE=""
if [[ "$FUSED_OK" == "1" && "$_PREFILL_FROM_ENV" == "0" && "$PREFILL_STEP" -lt 1024 ]]; then
  _PREFILL_NOTE="  (512 → 1024: fused aktiv, NAX braucht qL >= 1024)"
  PREFILL_STEP=1024
fi

# KV_BITS=8 war am 2026-08-21 kurzzeitig roomy-Default und ist wieder raus.
# Die Ueberlegung war: mit fused Attention faellt der Score-Transient weg, also
# dominieren die 64 KiB/Token, also halbieren. Der erste Teil stimmt, der
# Schluss nicht — apc_adapters.py:515 ruft beim Snapshot-Store
# dequantize_for_apc(), die APC-Snapshots liegen also weiter als f16 vor.
# Quantisiert wird nur der lebende Cache, und der ist nicht der Verbraucher.
# GEMESSEN: Decode 22,9 -> 18,7 tok/s (Mittel aus 6 bzw. 8 Requests), waehrend
# active unveraendert bis 37,78 GiB kletterte und derselbe OOM kam.
# Wer es trotzdem will:  KV_BITS=8 QUANT_KV_START=8192 ./start-mlx_qwen3.8.sh

# ── Drafter pruefen ───────────────────────────────────────────────────────────
# DFlash 2 haengt an zwei Patches. Fehlt 0040 (= Upstream-PR #1959), wuerde der
# Server beim Laden des Drafters hart abbrechen (unerwartete Gewichte) —
# deshalb hier abfangen und auf MTP zurueckfallen, statt den Start zu verlieren.
# Der Marker candidate_selector trifft sowohl den fruehreren eigenen Patch 0020
# als auch die Upstream-Fassung, die ihn ersetzt hat.
if [[ "$DRAFT_KIND" == "dflash" && "$ENABLE_SPEC_DECODE" != "0" ]]; then
  if ! grep -q "candidate_selector" \
       "$SITE_PACKAGES/mlx_vlm/speculative/drafters/qwen3_dflash/dflash.py" 2>/dev/null; then
    echo "⚠️  WARNUNG: Patch 0040 fehlt — DFlash 2 nicht ladbar, falle auf MTP zurueck." >&2
    echo "    Fix:  ./patches/apply-patches.sh" >&2
    DRAFT_KIND=mtp
    DRAFT_MODEL="$MODELS_ROOT/Qwen3.8-27B-MTP-4bit"
    DRAFT_BLOCK_SIZE=""
  elif ! grep -q "_speculative_batch_path_enabled" \
         "$SITE_PACKAGES/mlx_vlm/server/generation.py" 2>/dev/null; then
    echo "⚠️  WARNUNG: Patch 0021 fehlt — unter dflash greift dann KEIN Prefix-Cache" >&2
    echo "    (jeder Turn zahlt den vollen Prefill). Fix:  ./patches/apply-patches.sh" >&2
  fi
fi

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

apc_on = os.environ["ENABLE_APC"] == "1"
entries_req = int(os.environ["APC_ENTRIES"]) if apc_on else 0
model_ctx = int(os.environ.get("MODEL_CTX") or 0)
ctx_hint = int(os.environ.get("CTX_HINT") or 0)

# APC_EXACT_CACHE_ENTRIES deckelt die Zahl der Snapshots, nicht die Bytes — der
# Speicherbedarf ist also entries * Promptlaenge, unabhaengig von Patch 0010.
# Der Patch aendert nur, WIE VIELE KONVERSATIONEN in diese Eintraege passen
# (mit: eine pro Eintrag, ohne: eine pro zwei Eintraegen).
# ── Prefill-Transient ─────────────────────────────────────────────────────────
# Der bis 2026-08-21 fehlende Posten, und der Grund fuer "[METAL] Insufficient
# Memory" trotz scheinbar reichlichem Budget. Gilt nur noch OHNE Quellbau —
# mit aktivem Patch 0013 ist der Posten null (FUSED_OK-Zweig unten).
#
# head_dim ist 256. mlx' Default-Dispatch laesst fused Full-Attention nur fuer
# 64/80/128 zu (bis 0.32.1), die 16 Full-Attn-Layer laufen also auf dem unfused
# Graph und materialisieren je Layer einen Score-Tensor von
# n_heads x qL x kL x 2 B. Bei PREFILL_STEP 2048 und 38k Kontext sind das
# 2,8 GiB — PRO LAYER. Durch die verzoegerte Auswertung sind mehrere davon
# gleichzeitig lebendig.
#
# INFLIGHT = 16, die Zahl der Full-Attention-Layer: unter verzoegerter
# Auswertung kann im schlechtesten Fall jeder von ihnen seinen Score-Tensor
# gleichzeitig halten.
# GEGENGEPRUEFT auf 37,4 GiB Working-Set (kalter Prefill, Tier vorher geleert):
#   Chunk  512  gerechnet ~40k   gemessen bis ~38k durch, darueber OOM
#   Chunk 2048  gerechnet ~12k   gemessen bis ~30k durch
# Bei grossen Chunks ist die Rechnung also zu vorsichtig — das ist die richtige
# Fehlerrichtung. Unterschaetzen heisst hier Absturz mitten im Betrieb, und der
# kommt nicht als sauberer Fehler, sondern als
# "[METAL] Command buffer execution failed: Insufficient Memory".
# Faellt weg, sobald mlx 0.32.2 + Patch 0013 den fused Pfad liefern — genau das
# ist seit dem Source-Build vom 2026-08-21 der Fall, deshalb der FUSED_OK-Zweig.
# GEMESSEN mit mlx 0.32.2.dev+a082cb91, Produktionsform qL=512 / kL=22747,
# 24 Heads / 4 KV-Heads / head_dim 256:
#   force_fused=False : Peak 662 MiB   (Differenz 539 MiB = der Score-Tensor)
#   force_fused=True  : Peak 123 MiB
# Mit fused Pfad ist der Posten damit nicht "kleiner", sondern weg.
INFLIGHT = 16
n_heads = 24
prefill_step = int(os.environ.get("PREFILL_STEP") or 2048)
if os.environ.get("FUSED_OK") == "1":
    transient_per_tok = 0
else:
    transient_per_tok = INFLIGHT * n_heads * prefill_step * 2

def budget(entries):
    cop = 1 + entries                                    # 1 live + Snapshots
    av = ws - weights - reserve - cop * recurrent
    # Der Transient waechst mit der Kontextlaenge, nicht mit der Kopienzahl.
    tok = int(av / (cop * per_tok + transient_per_tok)) if av > 0 else 0
    if model_ctx:
        tok = min(tok, model_ctx)
    return cop, av, tok

# ── Ueberbuchung verhindern ───────────────────────────────────────────────────
# Ein Profil ist fuer einen bestimmten Working-Set gedacht (roomy fuer 44 GiB
# mit gesetztem wired_limit). Steht das Limit nicht, waehlt PROFILE=auto
# trotzdem roomy — und dann passt die Snapshot-Zahl nicht mehr zum Speicher.
# Symptom ist kein sauberer Fehler, sondern
# "[METAL] Command buffer execution failed: Insufficient Memory" mitten im
# Betrieb, oft erst nach mehreren Requests.
# Deshalb hier: die empfohlene Kontextlaenge muss mit MARGIN Luft ins Budget
# passen; sonst werden die Snapshots schrittweise reduziert. Gemessen auf einer
# 37,4-GiB-Maschine: mit 1,5 GiB Reserve starb jeder vierte Request, mit
# 7,5 GiB lief er durch.
MARGIN = 0.20                                            # 20 % Luft aufs Budget
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

# Hat die Budgetrechnung die Snapshot-Zahl gekappt, gilt der gekappte Wert.
if [[ "$ENABLE_APC" == "1" && "$ENTRIES_USED" != "$APC_ENTRIES" ]]; then
  echo "⚠️  APC_ENTRIES $APC_ENTRIES → $ENTRIES_USED gekappt: Profil '$PROFILE' ist fuer mehr" >&2
  echo "    Working-Set gedacht, als diese Maschine hat (${WS_GIB} GiB). Mit $APC_ENTRIES Snapshots" >&2
  echo "    haette context_length $_CTX_HINT keine Reserve — das endet im Betrieb in" >&2
  echo "    '[METAL] Insufficient Memory', nicht in einer sauberen Fehlermeldung." >&2
  echo "    Mehr Working-Set: sudo sysctl -w iogpu.wired_limit_mb=40960 (s. README)." >&2
  echo "    NICHT 45056 auf 48 GB, solange Browser/IDE mitlaufen — das laesst" >&2
  echo "    macOS ~2 GiB und endet in einer Kernel-Panik statt in Metal-OOM." >&2
  echo "    Ueberstimmen: APC_ENTRIES=$APC_ENTRIES ./start-mlx_qwen3.8.sh" >&2
  APC_ENTRIES="$ENTRIES_USED"
fi

# Score-Transient der unfused Full-Attention. Muss VOR dem Banner stehen und
# gehoert dort neben das KV-Budget: er waechst mit qL x kL, taucht in der
# KV-Rechnung nirgends auf, und ist ohne Patch 0013 die Groesse, die den
# Prefill zuerst gegen die Wand faehrt. Heads/Layer kommen aus der config.json,
# damit die Zahl auch nach einem Modellwechsel stimmt.
SCORE_GIB="—"
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
echo "  Mem-Probe:  $([[ "$_MEM_PROBE_INTERVAL" == "0" ]] && echo "OFF" || echo "alle ${_MEM_PROBE_INTERVAL}s ins Log (MEM_PROBE_INTERVAL=0 schaltet ab)")"
echo "  Prefill  :  Chunk $PREFILL_STEP   Slots: $MAX_NUM_SEQS   Vision-Cache: $VISION_CACHE$_PREFILL_NOTE"
echo "  Full-Attn:  $FUSED_STATUS"
echo "  Log      :  $LOG_FILE"
echo "  ──────────── Speicherbudget (gerechnet, keine Messung) ──────"
echo "  RAM              : ${RAM_GIB} GiB"
echo "  Metal-Working-Set: ${WS_GIB} GiB   (folgt iogpu.wired_limit_mb)"
echo "  Gewichte         : ${W_GIB} GiB   (+1,5 GiB Reserve fuer Aktivierungen)"
echo "  frei fuer KV     : ${AVAIL_GIB} GiB  auf ${COPIES} Kopien (1 live + APC)"
echo "  →  KONTEXT-BUDGET: ~${MAX_TOKENS_FIT} Token  — OBERGRENZE, keine Zusage"
echo "     Die Rechnung setzt (1+APC_ENTRIES) KV-Kopien an und unterschaetzt den"
echo "     realen Bedarf. Gemessen 2026-08-21: ein Request ueber 13k Token kostete"
echo "     +11,4 GiB statt der gerechneten 1,6, und active faellt zwischen den"
echo "     Requests nicht zurueck. Massgeblich sind die mem-Zeilen im Log."
if [[ "$FUSED_OK" == "0" ]]; then
echo "  Score-Transient  : ${SCORE_GIB} GiB  bei Chunk $PREFILL_STEP @ ${_CTX_HINT} Token"
echo "                     (unfused; kommt ZUSAETZLICH zum KV-Budget oben)"
fi
echo "──────────────────────────────────────────────────────────────"

# Harte Warnung, wenn das Working-Set auf dem 32-GB-Default steht.
if (( $(printf '%.0f' "$WS_GIB") < 24 )) && (( $(printf '%.0f' "$RAM_GIB") >= 30 )); then
  echo "ℹ️  Working-Set ist ${WS_GIB} GiB — das ist der macOS-Default (2/3 RAM)."
  echo "    Mit  sudo sysctl -w iogpu.wired_limit_mb=26624  werden daraus 26 GiB"
  echo "    und aus ~${MAX_TOKENS_FIT} Token rund das Doppelte. Persistent: siehe README."
fi
if [[ "$FUSED_OK" == "0" ]]; then
  echo "⚠️  Full-Attention laeuft unfused: $FUSED_STATUS"
  echo "    Der Prefill kippt dann in '[METAL] Insufficient Memory', lange bevor"
  echo "    das KONTEXT-BUDGET oben erreicht ist — der Score-Transient zaehlt mit."
  echo "    PREFILL_STEP NICHT erhoehen: der Transient waechst linear mit dem Chunk,"
  echo "    1024 verdoppelt ihn gegenueber 512. Erst mlx >= 0.32.2, dann der Chunk."
fi
if [[ -z "$KV_BITS" && "$MAX_TOKENS_FIT" -lt 40000 ]]; then
  echo "ℹ️  Unter 40k Token Budget. Verdoppeln ohne sudo:"
  echo "      KV_BITS=8 QUANT_KV_START=8192 ./start-mlx_qwen3.8.sh   (vorher A/B messen)"
fi
} | tee -a "$LOG_FILE"

# ── Client-Abgleich (optional, rein lesend) ───────────────────────────────────
# mlx-vlm hat KEIN -c-Flag; der Kontext kommt aus der config.json (262144). Der
# effektive Deckel ist also allein das context_length des Clients. Steht das
# ueber dem Budget oben, laeuft der Server irgendwann in
# "[METAL] Insufficient Memory".
#
# Wer eine Client-Konfiguration im YAML-Format mit einem model:-Block fuehrt
# (model.context_length, model.default), kann sie hier gegenpruefen lassen:
#
#   CLIENT_CONFIG=~/pfad/zu/config.yaml ./start-mlx_qwen3.8.sh
#
# Ohne die Variable ist der Block inert. Er aendert nichts, er warnt nur.
CLIENT_CONFIG="${CLIENT_CONFIG:-}"
if [[ -n "$CLIENT_CONFIG" && -f "$CLIENT_CONFIG" ]]; then
  CONFIG_CTX=$(awk '
    /^model:/ { in_model=1; next }
    in_model && /^[a-zA-Z_]/ { in_model=0 }
    in_model && /context_length:/ { gsub(/[^0-9]/,"",$0); print; exit }
  ' "$CLIENT_CONFIG")
  if [[ -n "$CONFIG_CTX" && "$CONFIG_CTX" -gt "$MAX_TOKENS_FIT" ]]; then
    echo "⚠️  WARNUNG: ${CLIENT_CONFIG:t} model.context_length=$CONFIG_CTX > Budget $MAX_TOKENS_FIT." >&2
    echo "    Der Client wird Prompts bauen, die hier nicht mehr in den Speicher passen." >&2
    echo "    Entweder context_length senken oder KV_BITS=8 / wired_limit anheben." >&2
  fi
  CONFIG_MODEL=$(awk '
    /^model:/ { in_model=1; next }
    in_model && /^[a-zA-Z_]/ { in_model=0 }
    in_model && /default:/ { sub(/^[^:]*:[[:space:]]*/,""); gsub(/["\x27]/,""); print; exit }
  ' "$CLIENT_CONFIG")
  if [[ -n "$CONFIG_MODEL" && "$CONFIG_MODEL" != "$MODEL_ALIAS" ]]; then
    echo "⚠️  WARNUNG: ${CLIENT_CONFIG:t} model.default='$CONFIG_MODEL' != Alias '$MODEL_ALIAS'." >&2
    echo "    Bei mlx-vlm IST der Request-Modellname der Ladepfad — Abweichung fuehrt" >&2
    echo "    zu Reload + HF-Download (401). Angleichen oder:" >&2
    echo "      MODEL_ALIAS='$CONFIG_MODEL' ./start-mlx_qwen3.8.sh" >&2
  fi
fi

# ── Thinking ──────────────────────────────────────────────────────────────────
# Serverseitig AUS (mlx-vlm schickt enable_thinking immer ans Template, Default
# false). Der Client setzt pro Request enable_thinking:true +
# reasoning_effort:low. GUELTIG SIND NUR low|medium|xhigh — jeder andere Wert,
# auch "none", wirft im Template eine Exception → HTTP 500. Ohne Angabe
# defaultet das Template auf 'xhigh' (gemessen 1269 statt 428 Completion-Tokens).
# Auf dieser Maschine ist 'low' nicht nur billiger, sondern ueberlebenswichtig:
# bei ~9 t/s sind 1269 Tokens ueber zwei Minuten reines Nachdenken.

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
  # Ohne das laufen Nicht-MTP-Drafter in einer eigenen Generierungsschleife, die
  # den APC-Manager nie verdrahtet: cached_tokens=0 in JEDEM Turn. Braucht
  # Patch 0021. Gemessen: cached 0 -> 5748/5788, Decode unveraendert.
  # Seit 2026-08-21 ueberschreibbar, damit der Pfad messbar ist. Er stand im
  # Verdacht, den fixen Speichersockel von ~9,4 GiB zu verursachen — GEMESSEN
  # IST ER ES NICHT: mit BATCH=0 bleibt der Sockel unveraendert bei +9,46 GiB.
  # Der Sockel haengt an --draft-block-size >= 2 (1 -> +0,16 GiB, 2 -> +9,31).
  # BATCH=0 kostet unter dflash den Prefix-Cache, ist also nichts fuer den
  # Produktivbetrieb — der Schalter existiert nur fuer Diagnoselaeufe.
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

# CWD = models-Root, damit der relative Alias-Name aufgeloest wird
# (get_model_path() macht Path(name).exists() gegen das Arbeitsverzeichnis).
cd "$MODELS_ROOT"

# ── Speicher-Sampler ──────────────────────────────────────────────────────────
# Warum ueberhaupt: mlx steckt jede Allokation in ein Residency-Set
# (mlx/backend/metal/allocator.cpp), macht sie also wired. Der Puffer-Cache wird
# bei gc_limit_ = 0,95 x Working-Set freigegeben, LEBENDER Speicher aber nie —
# fuer den gibt es unterhalb des Working-Sets keine Bremse. Ueberschreitet die
# Summe die Metal-Decke, scheitert der naechste Command Buffer mit
# "[METAL] Insufficient Memory", und zwar an einer beliebigen Stelle: am
# 2026-08-21 einmal in der Attention (ar.py:1907), einmal im APC-Klon
# (apc.py:321). Der Stacktrace zeigt deshalb den Ort, nicht die Ursache.
# Von aussen ist das nicht messbar: RSS enthaelt die Metal-Buffer nicht (2,3 GiB
# RSS bei 16 GiB Gewichten). Also von innen, per Thread, ohne mlx_vlm anzufassen
# — ein Patch in site-packages waere nach jedem mlx-vlm-Update wieder weg.
# Abschalten: MEM_PROBE_INTERVAL=0 (Wert wird oben bei LOG_PROGRESS gesetzt)
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
    # Erst schlafen: mlx_vlm konfiguriert das root-Logging beim Start, vorher
    # landet die Zeile im Nirwana statt im server.log.
    time.sleep(interval)
    while True:
        try:
            active = mx.get_active_memory()
            cache = mx.get_cache_memory()
            total = active + cache
            pct = (total / ws * 100.0) if ws else 0.0
            msg = (
                "mem active=%.2f cache=%.2f sum=%.2f GiB "
                "(%.0f%% von %.2f GiB Working-Set) peak=%.2f GiB"
            )
            argv = (
                active / _GIB, cache / _GIB, total / _GIB,
                pct, ws / _GIB, mx.get_peak_memory() / _GIB,
            )
            # Ab 85 % ist der naechste groessere Prefill der wahrscheinliche
            # Ausloeser — das soll im Log herausstechen, nicht in INFO untergehen.
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
