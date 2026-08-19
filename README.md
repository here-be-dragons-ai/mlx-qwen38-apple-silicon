# Qwen3.8-27B (MLX 4bit) auf Apple Silicon mit 32 GB

Setup- und Start-Skripte, um **Qwen3.8-27B** als lokalen, OpenAI-kompatiblen
Server (`mlx-vlm`) auf einem Mac mit **32 GB Unified Memory** zu betreiben —
inklusive der Speicher-Dimensionierung, die auf dieser Größe über
„läuft brauchbar" und „stirbt an `[METAL] Insufficient Memory`" entscheidet.

**Kurzantwort auf die Ausgangsfrage: ja, 27B dense läuft auf 32 GB.** Die
Gewichte (14,95 GiB) passen sogar in den macOS-Default. Das Nadelöhr ist der
KV-Cache mit **64 KiB pro Token** — der bestimmt, wieviel Kontext übrig bleibt
und damit, ob das Ding als Agent-Backend taugt.

Getestet auf macOS 26, Apple Silicon. Die Defaults sind auf 32 GB getunt; das
Start-Skript rechnet das Speicherbudget aber zur Laufzeit aus den echten Werten
der Maschine aus und funktioniert deshalb auf jedem Apple-Silicon-Mac.

---

## Voraussetzungen

| | |
|---|---|
| Hardware | Apple Silicon (arm64), ≥ 32 GB Unified Memory |
| macOS | aktuell, mit Xcode Command Line Tools (`xcode-select --install`) |
| Platte | ~20 GB für Modell + Drafter, dazu bis zu 40 GB für den SSD-Prefix-Cache |
| Netz | einmalig ~15 GB Download von HuggingFace |

Alles Weitere (uv, venv, Python 3.12, mlx-vlm) installiert `install-prereqs.sh`.

---

## Installation

```sh
git clone https://github.com/here-be-dragons-ai/mlx-qwen38-m5-32gb.git
cd mlx-qwen38-m5-32gb

# 1. Software + Modellgewichte (idempotent, Downloads sind resume-fähig)
./install-prereqs.sh

# 2. GPU-Wired-Limit anheben — der wichtigste Schritt auf 32 GB (s.u.)
sudo sysctl -w iogpu.wired_limit_mb=26624

# 3. Server starten (127.0.0.1:8888)
./start-mlx_qwen3.8.sh
```

`install-prereqs.sh` legt an bzw. prüft: Xcode CLT → [uv](https://astral.sh/uv)
→ venv unter `~/src/mlx/.venv` (Python 3.12) → mlx-vlm + Abhängigkeiten →
Metal-Selbsttest → Patches → Modell + MTP-Drafter → `~/.hermes/{logs,apc}`.

| Option | Wirkung |
|---|---|
| `--check` | nur prüfen, nichts verändern |
| `--skip-model` | Software ja, 15-GB-Download nein |
| `--latest` | neueste statt der gepinnten Versionen |

Pfade sind über Env steuerbar: `MLX_HOME` (Default `~/src/mlx`), `MLX_MODELS`,
`PYTHON_VERSION`.

**Gepinnter, verifizierter Stand:** `mlx 0.32.0`, `mlx-lm 0.31.3`,
**`mlx-vlm 0.6.13`**, `transformers 5.15.0`, `numpy 2.5.2`,
`huggingface-hub 1.27.0`, `pillow 12.3.0`, Python 3.12.
`mlx-vlm >= 0.6.13` ist Pflicht — erst dort ist Prefix-Caching upstream korrekt
(`semantic_extra_hash()`); davor trifft der Cache nur bei byte-identischen Prompts.

### Wired-Limit persistent machen

`sysctl -w` überlebt keinen Neustart. Dauerhaft:

```sh
sudo cp com.local.iogpu-wired-limit.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/com.local.iogpu-wired-limit.plist
sudo launchctl load -w /Library/LaunchDaemons/com.local.iogpu-wired-limit.plist
```

---

## Betrieb

```sh
./start-mlx_qwen3.8.sh                                    # Defaults (32 GB)
KV_BITS=8 QUANT_KV_START=8192 ./start-mlx_qwen3.8.sh      # doppelter Kontext
PORT=8899 ./start-mlx_qwen3.8.sh                          # Laborinstanz
ENABLE_SPEC_DECODE=0 ./start-mlx_qwen3.8.sh               # ohne Drafter
```

Beim Start druckt das Skript das errechnete Speicherbudget dieser Maschine:

```
  ──────────── Speicherbudget (gerechnet, keine Messung) ──────
  RAM              : 32 GiB
  Metal-Working-Set: 26.0 GiB   (folgt iogpu.wired_limit_mb)
  Gewichte         : 15.2 GiB   (+1,5 GiB Reserve fuer Aktivierungen)
  frei fuer KV     : 8.9 GiB  auf 3 Kopien (1 live + APC)
  →  KONTEXT-BUDGET: ~48486 Token pro Konversation
```

Wichtigste Env-Schalter (alle mit Begründung im Skriptkopf dokumentiert):

| Variable | Default | Wirkung |
|---|---|---|
| `KV_BITS` | leer (f16) | `8` halbiert 64 → 32 KiB/Token, verdoppelt den Kontext |
| `APC_ENTRIES` | `2` | Prefix-Cache-Snapshots = warm gehaltene Konversationen |
| `ENABLE_SPEC_DECODE` | `1` | MTP-Drafter, +58…132 % Decode |
| `PREFILL_STEP` | `1024` | Prefill-Chunk = transienter Aktivierungs-Peak |
| `BIND_HOST` / `PORT` | `127.0.0.1` / `8888` | Bind-Adresse |
| `MODEL_ALIAS` | `Qwen3.8-27B-local` | **muss** zum Modellnamen im Request passen |

> Bei `mlx-vlm` **ist** der `model`-String aus dem Request der Ladepfad — es gibt
> kein `--alias` wie bei llama.cpp. Weicht der Name ab, wirft der Server das
> geladene Modell weg und startet einen HuggingFace-Download (→ 401, obwohl das
> Modell lokal liegt). Das Skript legt dafür einen Symlink an und warnt bei
> Abweichung.

---

## Die Speicherrechnung

| Posten | Größe | Herkunft |
|---|---|---|
| Gewichte, 4bit affine (3 Shards) | **14,95 GiB** | Dateigröße |
| MTP-Drafter (Speculative Decoding) | 0,23 GiB | Dateigröße |
| Aktivierungen, Metal-Heap, Python | ~1,5 GiB | Erfahrungswert |
| KV-Cache | **64 KiB / Token** | 16 Full-Attn-Layer × 4 KV-Heads × (256+256) × 2 B |
| GDN-State (48 Linear-Layer) | 152 MiB, **längenunabhängig** | 48 × 48 v-Heads × 128 × 128 × 4 B (float32) |

Qwen3.8-27B ist ein **Hybrid**: nur 16 der 64 Layer sind Full-Attention (die
zahlen KV pro Token), die anderen 48 sind Gated DeltaNet mit konstantem
rekurrentem State. Und es ist **dense** — kein MoE, jeder Token liest alle
~15 GiB.

Der KV-Posten fällt **pro Kopie** an: einmal für die laufende Sequenz, einmal
pro Prefix-Cache-Snapshot. Mit `APC_ENTRIES=2` sind das 3 Kopien:

```
Budget = (Working-Set − Gewichte − 1,5 GiB − Kopien × 152 MiB) / (Kopien × KV_pro_Token)
```

Metals `max_recommended_working_set_size` ist auf Macs **≤ 36 GB per Default
2/3 des RAM** → auf 32 GB nur 21,33 GiB. `iogpu.wired_limit_mb` setzt diesen
Wert direkt (verifiziert: `wired_limit_mb=45056` → Working-Set exakt 44 GiB).

| `iogpu.wired_limit_mb` | KV | Budget | empfohlenes `context_length` |
|---|---|---|---|
| Default (21,3 GiB) | f16 | ~23k Token | 24576 — knapp, funktioniert |
| Default (21,3 GiB) | `KV_BITS=8` | ~46k Token | 32768 |
| **26624 (26 GiB)** | **f16** | **~48k Token** | **49152 ← Empfehlung** |
| 26624 (26 GiB) | `KV_BITS=8` | ~97k Token | 65536 (bis 98304) |

**Empfehlung: Wired-Limit 26 GiB, KV f16, Kontext 49152.** Damit ist kein
einziger ungemessener Trade-off im Spiel.

Warum nicht höher als 26624: macOS braucht selbst 5–6 GB. Ein zu hohes Limit
tauscht „Metal OOM" gegen Beachball bzw. Kernel-Panic. Reine
Inferenz-Maschine ohne Browser/IDE: 28672 geht.

---

## Client-Konfiguration

Der Server spricht OpenAI-Chat-Completions auf `http://localhost:8888/v1`.
Beispiel für [Hermes](https://github.com/gtonic/hermes) (`~/.hermes/config.yaml`);
für andere Clients gelten dieselben zwei Regeln: **Modellname = Alias** und
**Kontext ≤ Budget**.

```yaml
model:
  default: Qwen3.8-27B-local      # MUSS zum Symlink-Namen passen
  provider: custom:llama-local
  base_url: http://localhost:8888/v1
  api_key: sk-local
  api_mode: chat_completions
  context_length: 49152            # ← 32-GB-Wert
  max_tokens: 8192                 # NICHT erhöhen: Kompaktierungs-Trigger ist
  supports_vision: true            # (context_length − max_tokens) × threshold
  extra_body:
    enable_thinking: true
    reasoning_effort: low          # nur low|medium|xhigh — alles andere → HTTP 500
compression:
  threshold: 0.85
```

Rechnung: (49152 − 8192) × 0,85 → Kompaktierung bei ~34,8k, Spitzenprompt ~43k,
unter dem Budget von ~48k. Das Start-Skript warnt, wenn `context_length` über
dem errechneten Budget liegt.

**Thinking:** Das Chat-Template von Qwen3.8 akzeptiert nur `low|medium|xhigh`
und wirft bei jedem anderen Wert — auch `none` — eine Exception (HTTP 500).
Ohne Angabe defaultet es auf `xhigh`, die teuerste Stufe (gemessen: 1269 statt
428 Completion-Tokens). Bei ~9 t/s sind das zwei Minuten Unterschied. Steigt
die Tool-Fehlerrate, auf `medium` gehen — zu flaches Reasoning kann die
Gesamtlatenz durch Fehlversuche *erhöhen*.

---

## Zu erwartende Geschwindigkeit

Dense heißt: jeder Decode-Schritt liest ~15 GiB, das ist reine
Speicherbandbreite. Gemessen auf einem M5 Pro, hochskaliert auf die
bandbreitenärmere Basis-Variante (~153 GB/s):

| | M5 Pro / 48 GB (gemessen) | M5 Basis / 32 GB (**geschätzt**) |
|---|---|---|
| Decode roh | 17,5–18,4 t/s | ~8–10 t/s |
| Decode mit MTP-SpecDec (Tool-Calls/JSON) | 26,9–41,5 t/s | ~13–20 t/s |
| Prefill | 420–470 t/s | ~180–250 t/s |

Ein **kalter** 30k-Prefill dauert damit 2–3 Minuten. Deshalb sind Prefix-Cache
und SSD-Tier hier keine Optimierung, sondern Voraussetzung: gemessen 89 630 ms
→ 350 ms für einen 36k-Prompt nach Serverneustart (Faktor 256).

---

## Wenn es zu eng wird

In dieser Reihenfolge:

1. **`KV_BITS=8 QUANT_KV_START=8192`** — verdoppelt den Kontext, lässt die
   ersten 8k unquantisiert. Vorher A/B messen: bei llama.cpp kostete
   KV-Quantisierung an vergleichbarer Stelle bis zu 8× Prefill und 1,9× Decode;
   für den MLX-Pfad ist das *nicht* nachgemessen.
2. **`APC_ENTRIES=1`** — nur eine Konversation warm, dafür 50 % mehr Kontext.
3. **Kleineres Quant** (3bit/DWQ, falls verfügbar) — spart 3–4 GiB, kostet
   Qualität: `./download-mlx-model.sh <repo> <ziel>` und `MODEL_DIR=… ./start-…`.
4. **Kleineres Modell.** Ab hier ist die ehrliche Antwort, dass 27B dense auf
   32 GB einfach knapp ist.

**Nicht** `--max-kv-size` benutzen (rotierendes KV-Fenster): dieselbe Idee wurde
mit einem Needle-in-Haystack-Test widerlegt — die Nadel außerhalb des Fensters
ging verloren und wurde in einem Fall sogar halluziniert (8347 statt 8342).
Stiller Qualitätsverlust ist schlimmer als ein sauberer kleiner Kontext.

---

## Diagnose

```sh
# Trifft der Prefix-Cache? (Turn 2 muss cached_tokens > 0 zeigen)
grep -o 'cached[_ ]tokens[=:] *[0-9]*' ~/.hermes/logs/mlx-qwen3.8.log | tail -20

# Draft-Acceptance (unter ~40 % lohnt Speculative Decoding nicht)
grep -i "accept" ~/.hermes/logs/mlx-qwen3.8.log | tail -10

# Speicherlage
~/src/mlx/.venv/bin/python -c "import mlx.core as mx;print(mx.device_info())"
sysctl iogpu.wired_limit_mb
ps -o rss=,command= -p "$(pgrep -f mlx_vlm.server)" | awk '{printf "%.1f GiB\n", $1/1048576}'

# Patch-Status (Patches liegen in site-packages und verschwinden bei JEDEM pip install)
./patches/apply-patches.sh --check
```

| Symptom | Ursache |
|---|---|
| `cached_tokens=1` bei großen Prompts | Patch 0002 fehlt |
| `cached_tokens=0` in Turn 2 | mlx-vlm < 0.6.13, oder Snapshot verdrängt (`APC_ENTRIES`, Patch 0010) |
| HTTP 401 / HF-Download beim Request | Modellname ≠ Alias-Symlink |
| HTTP 500 bei jedem Request | `reasoning_effort` außerhalb `low\|medium\|xhigh` |
| `[METAL] Insufficient Memory` | Kontext über Budget → `context_length` senken oder `KV_BITS=8` |

---

## Patches

Zwei Patches gegen `site-packages`, angewendet von `patches/apply-patches.sh`
(idempotent, `--check` / `--revert`). Sie verschwinden bei jedem
`pip install -U mlx-vlm` — danach erneut ausführen.

- **`0002-pr1901-apc-short-prompt.patch`** — Upstream-PR #1901, gemerged *drei
  Tage nach* dem 0.6.13-Release und deshalb nicht im PyPI-Stand. Ohne ihn
  deaktiviert ein einziger kurzer Prompt den Prefix-Cache für den ganzen
  Tenant (Signatur: `cached_tokens=1` bei großen Prompts). Gemessene Kosten in
  einem realen Log: 13 Requests = 1055 s verlorene Prefill-Zeit, ausgelöst von
  6 kurzen Prompts. Meldet `apply-patches.sh` hier „KONFLIKT", ist der Fix
  upstream angekommen → Datei löschen.
- **`0010-qwen38-apc-single-snapshot.patch`** — lokal, kein Upstream-PR.
  mlx-vlm legt pro Request zwei fast identische Snapshots ab (Checkpoint bei
  `len-16` und den vollen Prompt); getroffen wird gemessen immer der
  Checkpoint. Der Patch unterdrückt den zweiten und halbiert damit den
  Cache-Speicher. Ohne die Env-Variable `QWEN38_APC_SINGLE_SNAPSHOT=1` inert —
  das ist der Rollback-Pfad.

---

## Dateien

| Datei | Zweck |
|---|---|
| `install-prereqs.sh` | Komplettes Setup ab frischem macOS, idempotent |
| `start-mlx_qwen3.8.sh` | Server-Start, 32-GB-Defaults, Live-Budgetrechnung |
| `download-mlx-model.sh` | Resume-fähiger HuggingFace-Downloader (curl, mit Größenprüfung) |
| `patches/apply-patches.sh` | Patches anwenden / prüfen / zurücknehmen |
| `com.local.iogpu-wired-limit.plist` | LaunchDaemon für das Wired-Limit |

---

## Herkunft der Zahlen

Alle mit „gemessen" markierten Werte stammen von einer **M5 Pro / 48 GB**
Maschine (mlx-vlm 0.6.13, Qwen3.8-27B-4bit, `temperature=0`). Übernommen sind
nur die hardwareunabhängigen Erkenntnisse:

- MTP-Speculative-Decoding lohnt (Decode +58…132 %, Acceptance 42 % Prosa /
  90 % JSON / 93 % Tool-Call, Qualität 7/7 bit-identisch)
- Prefix-Caching + SSD-Tier lohnen (Faktor 256 auf einen kalten 36k-Prefill)
- KV-Fensterung ist widerlegt (Halluzination außerhalb des Fensters)

**Die Speicherrechnung ist Arithmetik** aus `config.json` und den
Dateigrößen — die gilt auf jeder Maschine. **Die Durchsatzangaben für 32 GB
sind Schätzungen**, hochskaliert über die Speicherbandbreite, und als solche
gekennzeichnet.
