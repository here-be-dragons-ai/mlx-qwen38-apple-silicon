# Qwen3.8-27B (MLX 4bit) auf Apple Silicon

Setup- und Start-Skripte, um **Qwen3.8-27B** als lokalen, OpenAI-kompatiblen
Server (`mlx-vlm`) auf einem Apple-Silicon-Mac zu betreiben — inklusive der
Speicher-Dimensionierung, die über „läuft brauchbar" und „stirbt an
`[METAL] Insufficient Memory`" entscheidet.

Die Gewichte (14,95 GiB) sind dabei **nicht** das Problem. Das Nadelöhr ist der
KV-Cache mit **64 KiB pro Token**, der pro Kopie anfällt (laufende Sequenz +
jeder Prefix-Cache-Snapshot). Er bestimmt, wieviel Kontext übrig bleibt — und
damit, ob das Ding als Agent-Backend taugt.

Drei Profile decken die üblichen Maschinen ab; `PROFILE=auto` (Default) wählt
selbst:

| Profil | Zielmaschine | Peak-RAM | `context_length` |
|---|---|---|---|
| `lean` | 32 GB **ohne** `sudo` | ~18,8 GiB | 32768 |
| `balanced` | 32 GB mit `wired_limit 26624` | ~25,0 GiB | 49152 |
| `roomy` | 48 GB mit `wired_limit 45056` | ~41 GiB | 98304 |

Getestet auf macOS 26. `roomy` ist das ursprüngliche, über Wochen produktiv
gefahrene M5-Pro-Setup; die 32-GB-Profile sind daraus gerechnet. Das
Start-Skript ermittelt das Speicherbudget zur Laufzeit aus den echten Werten
der Maschine und warnt, wenn die Client-Konfiguration darüber liegt.

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
git clone https://github.com/here-be-dragons-ai/mlx-qwen38-apple-silicon.git
cd mlx-qwen38-apple-silicon

# 1. Software + Modellgewichte (idempotent, Downloads sind resume-fähig)
./install-prereqs.sh

# 2. GPU-Wired-Limit anheben — der wichtigste Schritt (s.u.)
sudo sysctl -w iogpu.wired_limit_mb=26624    # 32 GB;  48 GB → 45056

# 3. Server starten (127.0.0.1:8888)
./start-mlx_qwen3.8.sh
```

Schritt 2 ist auf 32 GB der Unterschied zwischen ~23k und ~48k nutzbarem
Kontext. Ohne ihn läuft das Setup trotzdem — `PROFILE=auto` erkennt das und
schaltet auf `lean`.

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
**`mlx-vlm 0.6.15`**, `transformers 5.15.0`, `numpy 2.5.2`,
`huggingface-hub 1.27.0`, `pillow 12.3.0`, Python 3.12.
`mlx-vlm >= 0.6.13` ist Pflicht — erst dort ist Prefix-Caching upstream korrekt
(`semantic_extra_hash()`); davor trifft der Cache nur bei byte-identischen
Prompts. Ab `0.6.14` ist zusätzlich der Kurzprompt-Fix (PR #1901) enthalten,
der vorher als lokaler Patch nötig war.

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
./start-mlx_qwen3.8.sh                        # PROFILE=auto
PROFILE=lean ./start-mlx_qwen3.8.sh           # minimaler RAM, läuft ohne sudo
PROFILE=roomy ./start-mlx_qwen3.8.sh          # 48-GB-Setup
PORT=8899 ./start-mlx_qwen3.8.sh              # Laborinstanz
ENABLE_SPEC_DECODE=0 ./start-mlx_qwen3.8.sh   # ohne Drafter
```

### Profile

| | `lean` | `balanced` | `roomy` |
|---|---|---|---|
| Zielmaschine | 32 GB ohne sudo | 32 GB, `wired_limit 26624` | 48 GB, `wired_limit 45056` |
| `APC_ENTRIES` | 1 | 2 | 4 |
| `KV_BITS` | 8 (ab 8k Token) | — (f16) | — (f16) |
| `PREFILL_STEP` | 512 | 1024 | 2048 |
| `VISION_CACHE` | 1 | 4 | 20 |
| `APC_DISK_MAX_GB` | 40 | 40 | 60 |
| **Peak-RAM** | **~18,8 GiB** @ 29k | ~25,0 GiB @ 43k | ~41 GiB @ 90k |
| `context_length` | 32768 | 49152 | 98304 |

**`PROFILE=auto`** (Default) entscheidet anhand des Metal-Working-Sets:
≥ 30 GiB → `roomy`, ≥ 24 GiB → `balanced`, sonst `lean`. Der Working-Set wird
dafür aus `iogpu.wired_limit_mb` bzw. dem macOS-Default (2/3 des RAM bei
≤ 36 GB, sonst 3/4) geschätzt — die exakte Zahl aus Metal steht danach im
Banner. `default` bleibt als Alias für `balanced` erhalten.

Ein Profil setzt nur *Defaults* — einzelne Env-Variablen gewinnen weiterhin,
z. B. `PROFILE=roomy APC_ENTRIES=2 …` oder `PROFILE=lean KV_BITS=` (zurück auf
f16).

> **Warum `roomy` 98304 empfiehlt, obwohl das 48-GB-Setup produktiv mit 131072
> lief:** die Budgetrechnung ist ein Worst Case — alle vier Snapshots
> gleichzeitig auf voller Promptlänge. Bei 44 GiB Working-Set und 5 Kopien sind
> das ~87k Token. Dass 131072 in der Praxis trug, liegt daran, dass die
> Snapshots real nie alle gleichzeitig am Anschlag stehen; es ist Glück, keine
> Garantie. Wer 131072 fahren will, nimmt `APC_ENTRIES=2` dazu (3 Kopien →
> Budget ~145k).

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
| `PROFILE` | `auto` | `lean` / `balanced` / `roomy` (Tabelle oben) |
| `KV_BITS` | profilabhängig | `8` halbiert 64 → 32 KiB/Token, verdoppelt den Kontext |
| `APC_ENTRIES` | profilabhängig | Prefix-Cache-Snapshots = warm gehaltene Konversationen |
| `ENABLE_SPEC_DECODE` | `1` | MTP-Drafter, +58…132 % Decode |
| `PREFILL_STEP` | profilabhängig | Prefill-Chunk = transienter Aktivierungs-Peak |
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
Bedarf = 16,7 GiB + Kopien × (Token × KV_pro_Token + 152 MiB)
```

### Wieviel RAM braucht es konkret?

Profil `default`, ctx 49152:

| Zustand | RAM |
|---|---|
| Gewichte geladen, Leerlauf | 15,2 GiB (gemessen: RSS 15,5 GiB) |
| + Aktivierungen / Metal-Heap / Python | ~16,7 GiB |
| Betrieb, 8k-Prompt | ~18,7 GiB |
| Betrieb, Spitzenprompt 43k | **~25,0 GiB** ← dimensionierend |

Profil `lean` mit ctx 32768 kommt auf **~18,8 GiB** Peak. Der absolute Boden für
dieses Modell liegt bei **~17 GiB** (ohne Drafter, ohne Cache, 8k Kontext, KV4)
— darunter hilft nur ein anderes Modell oder ein kleineres Quant.

Dazu kommen 5–6 GB für macOS selbst; das ist der Grund, warum auf 32 GB bei
26 GiB Wired-Limit Schluss ist.

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
  context_length: 49152            # ← Profil balanced (lean 32768, roomy 98304)
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

Erster Griff ist `PROFILE=lean`. Was dahintersteckt, einzeln und mit Preis
(Ersparnis bezogen auf einen 43k-Spitzenprompt):

| Hebel | spart | Preis |
|---|---|---|
| `APC_ENTRIES=1` | **−2,8 GiB** | gering — der SSD-Tier bleibt, ein verdrängter Snapshot ist in 350 ms zurück statt in 90 s Vollprefill |
| `KV_BITS=8 QUANT_KV_START=8192` | **−4,2 GiB** | Durchsatz auf dem MLX-Pfad *nicht* gemessen; bei llama.cpp kostete KV-Quantisierung an vergleichbarer Stelle bis 8× Prefill und 1,9× Decode → A/B messen |
| `context_length` 49152 → 32768 | −2,6 GiB | kürzere Läufe bis zur Kompaktierung |
| `PREFILL_STEP=512` | ~−0,2 GiB | praktisch keiner (Prefill ist rechen-, nicht chunklimitiert) |
| Gewichte `mxfp4` statt `4bit` | −0,78 GiB | `mlx-community/Qwen3.8-27B-mxfp4` = 14,17 statt 14,95 GiB, von mlx 0.32 unterstützt; Qualität unverglichen |
| Drafter weglassen | −0,23 GiB | **schlechter Tausch** — kostet 58–132 % Decode |
| `ENABLE_APC=0` | −5,5 GiB | **inakzeptabel** — jeder Turn zahlt den vollen Prefill |

Zwei Dinge, die man beim Suchen findet und die nichts bringen:

- **Der Vision-Tower ist 0,86 GiB und liegt unquantisiert in BF16 im Modell**
  (5,7 % der Gewichte; die 4bit-Quantisierung hat ihn übersprungen). Für einen
  reinen Text-Agenten totes Gewicht — aber mlx-vlm 0.6.13 wertet den
  `language_model_only`-Schalter aus der `config.json` nicht aus (kein Treffer
  im Code). Es gibt also keinen Schalter; nur Strippen/Requantisieren der
  Gewichte von Hand.
- **Unter 4bit gibt es nichts Fertiges.** mlx-community führt für Qwen3.8-27B
  4bit, 8bit, mxfp4, nvfp4, oQ4/oQ6 und OptiQ — alles ≥ 14,17 GiB, kein 3bit,
  kein DWQ. Selbst quantisieren ginge (`mlx_vlm.convert -q --q-bits 3`, ~11,5 GiB),
  ist aber ein Qualitätsexperiment mit offenem Ausgang.

Ab hier ist die ehrliche Antwort, dass 27B dense auf 32 GB knapp ist und ein
kleineres Modell die bessere Wahl wäre.

**Nicht** `--max-kv-size` benutzen (rotierendes KV-Fenster): dieselbe Idee wurde
mit einem Needle-in-Haystack-Test widerlegt — die Nadel außerhalb des Fensters
ging verloren und wurde in einem Fall sogar halluziniert (8347 statt 8342).
Stiller Qualitätsverlust ist schlimmer als ein sauberer kleiner Kontext.

---

## DFlash2 — portiert, gemessen, aber nicht Default

[DFlash 2](https://inco.ai/blog/dflash2/) ist als Drafter für Qwen3.8-27B
portiert (`patches/0020-dflash2-qwen38.patch`) und über `DRAFT_KIND=dflash`
nutzbar. **Default bleibt trotzdem der MTP-Drafter** — der Grund steht unten.

Hintergrund: mlx-vlm bringt zwar die Drafter-Klasse `qwen3_dflash` und die
Zielmodell-Hooks mit, implementiert aber nur DFlash **v1**; die v2-Neuerungen
(Kandidaten-Pfad-Selektor, Two-Tap-Dynamic-Convolutions) fehlen in 0.6.13,
0.6.15 und auf upstream `main`. Auch oMLX 0.6.2 (`dflash_mlx 0.1.10+omlx.5`) hat
nur v1. Für Qwen3.8-27B existiert aber gar kein v1-Drafter — nur der v2. Der
Patch schließt genau diese Lücke, transkribiert aus der MLX-Referenz von z-lab
([`dflash/model_mlx.py`](https://github.com/z-lab/dflash/blob/main/dflash/model_mlx.py)).

### Verwendung

```sh
./download-mlx-model.sh z-lab/Qwen3.8-27B-DFlash2 ~/src/mlx/models/Qwen3.8-27B-DFlash2-bf16
./convert-dflash2-drafter.py ~/src/mlx/models/Qwen3.8-27B-DFlash2-bf16 \
                             ~/src/mlx/models/Qwen3.8-27B-DFlash2-4bit
DRAFT_KIND=dflash ./start-mlx_qwen3.8.sh          # block_size 4 (Default)
```

### Was gemessen wurde (M5 Pro, mlx-vlm 0.6.15, identische Prompts)

| Prompt | MTP | DFlash2 (block 4) | |
|---|---|---|---|
| 64 Token | 33,9 t/s | **40,8 t/s** | +20 % |
| 66 Token | 33,9 t/s | **38,2 t/s** | +13 % |
| 76 Token | 36,7 t/s | **45,6 t/s** | +24 % |
| 5767 Token | 33,9 t/s | **38,4 t/s** | +13 % |

Blockgrößen-Sweep (Mittel der drei kurzen Prompts, gegen MTP):
`block 3` +6 %, **`block 4` +19 %**, **`block 5` +20 %**, `block 8` +6 %.
Der Checkpoint ist auf `block_size 8` ausgelegt — auf einem 4bit-Target ist das
gemessen die *schlechteste* Wahl, genau wie z-lab es für quantisierte
MLX-Modelle ankündigt (`block_size <= 5`). Acceptance bei Block 4–5:
3,0–3,7 angenommene Token pro Runde.

Korrektheit: Ausgabe bei `temperature 0` **4/4 bit-identisch zum MTP-Drafter**,
Tool-Call korrekt. Die portierten Module sind gegen die Referenz geprüft —
81/81 Parameter in Name und Form, Conv `max|diff| = 0`, Selector-Pfade identisch.

### Der Prefix-Cache greift jetzt auch unter DFlash2

Ursprünglich meldete unter `DRAFT_KIND=dflash` **jeder** Request
`cached_tokens=0`. Ursache gefunden: `server/generation.py` routet jeden
Nicht-MTP-Drafter in eine zweite Generierungsschleife (`_run_speculative`), die
ihren eigenen Prompt-Cache baut und den APC-Manager nie verdrahtet. Der
Continuous-Batching-Pfad kann dflash längst — er ist durchgehend generisch über
`draft_kind` und bekommt `apc_manager`, `draft_kind` und `draft_block_size` in
derselben Zeile übergeben. Nur die Weiche hielt dflash davon fern.

`patches/0021-speculative-apc-routing.patch` macht den Batch-Pfad über
`MLX_VLM_SPECULATIVE_BATCH=1` erreichbar; das Start-Skript setzt die Variable
automatisch, sobald `DRAFT_KIND != mtp`. Gemessen (5,8k-Konversation, Turn 2):

| | `cached_tokens` | Decode 64/66/76 Tok | 5767 Tok |
|---|---|---|---|
| MTP | 5772 / 5788 | 33,9 / 33,9 / 36,7 t/s | 33,9 t/s |
| DFlash2, alte Schleife | **0** | 40,8 / 38,2 / 45,6 t/s | 38,4 t/s |
| DFlash2, Batch-Pfad | **5748 / 5788** | 38,7 / 40,7 / 43,0 t/s | **40,9 t/s** |

Der Durchsatz bleibt also gleich (im Mittel 40,8 statt 41,5 t/s — Rauschen) und
wird beim langen Prompt sogar besser, aber der Prefix-Cache ist wieder da.
`--draft-block-size` wirkt weiterhin (Block 4 schlägt Block 8 auf beiden Pfaden),
zwei parallele Requests mit `MAX_NUM_SEQS=2` laufen sauber, und MTP bleibt
unverändert (`cached=5772`).

### Warum MTP trotzdem noch Default ist

Kein technischer Grund mehr — ein betrieblicher: DFlash2 hängt an **zwei**
lokalen Patches (0020 für die Drafter-Module, 0021 fürs Routing). Ein
`pip install -U mlx-vlm` ohne anschließendes `apply-patches.sh` würde den
Drafter unladbar machen. Das Start-Skript fängt das ab — es prüft beide Patches
und fällt notfalls mit Warnung auf MTP zurück — aber MTP läuft hier seit Wochen
produktiv, DFlash2 seit Stunden.

Empfehlung: ein paar Tage mit `DRAFT_KIND=dflash` fahren, dann den Default
umstellen (eine Zeile im Start-Skript). Beide Patches sind als PR bei mlx-vlm
eingereicht; sobald sie upstream sind, entfällt die Patch-Abhängigkeit.

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
| `cached_tokens=1` bei großen Prompts | mlx-vlm < 0.6.14 (Kurzprompt-Bug, PR #1901) |
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
| `LICENSE` | MIT No Attribution (SPDX `MIT-0`) |
| `start-mlx_qwen3.8.sh` | Server-Start, Profile lean/balanced/roomy, Live-Budgetrechnung |
| `download-mlx-model.sh` | Resume-fähiger HuggingFace-Downloader (curl, mit Größenprüfung) |
| `patches/apply-patches.sh` | Patches anwenden / prüfen / zurücknehmen |
| `com.local.iogpu-wired-limit.plist` | LaunchDaemon für das Wired-Limit |

---

## Herkunft der Zahlen

Alle mit „gemessen" markierten Werte stammen von einer **M5 Pro / 48 GB**
Maschine (mlx-vlm 0.6.13/0.6.15, Qwen3.8-27B-4bit, `temperature=0`). Übernommen sind
nur die hardwareunabhängigen Erkenntnisse:

- MTP-Speculative-Decoding lohnt (Decode +58…132 %, Acceptance 42 % Prosa /
  90 % JSON / 93 % Tool-Call, Qualität 7/7 bit-identisch)
- Prefix-Caching + SSD-Tier lohnen (Faktor 256 auf einen kalten 36k-Prefill)
- KV-Fensterung ist widerlegt (Halluzination außerhalb des Fensters)

**Die Speicherrechnung ist Arithmetik** aus `config.json` und den
Dateigrößen — die gilt auf jeder Maschine. **Die Durchsatzangaben für 32 GB
sind Schätzungen**, hochskaliert über die Speicherbandbreite, und als solche
gekennzeichnet.

---

## Lizenz

[MIT No Attribution](LICENSE) (SPDX: `MIT-0`) — MIT ohne die Pflicht, den
Copyright-Vermerk weiterzugeben. Kopieren, anpassen, weiterverwenden ohne
Namensnennung.
