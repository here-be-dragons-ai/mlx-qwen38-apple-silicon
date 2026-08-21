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
| `roomy` | 48 GB mit `wired_limit 45056` | ~28 GiB | 98304 |

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

**Gepinnter, verifizierter Stand:** `mlx 0.32.1`, `mlx-lm 0.31.3`,
**`mlx-vlm 0.6.15`**, `transformers 5.15.0`, `numpy 2.5.2`,
`huggingface-hub 1.27.0`, `pillow 12.3.0`, Python 3.12.
`mlx-vlm >= 0.6.13` ist Pflicht — erst dort ist Prefix-Caching upstream korrekt
(`semantic_extra_hash()`); davor trifft der Cache nur bei byte-identischen
Prompts. Ab `0.6.14` ist zusätzlich der Kurzprompt-Fix (PR #1901) enthalten,
der vorher als lokaler Patch nötig war.

`mlx 0.32.1` ist ein reiner Kompatibilitätsschritt und **kostet nichts**.
Gemessen auf dieser Maschine (12 Läufe je Version, `temperature 0`, feste
Prompts, Decode-Rate aus dem Server-Log):

| | Median | Mittel | min–max |
|---|---|---|---|
| `mlx 0.32.0` | 43,2 t/s | 43,9 t/s | 42,8–45,4 |
| `mlx 0.32.1` | 43,3 t/s | 43,3 t/s | 40,7–47,2 |

Gleiche Token-Zahl in beiden Läufen, und die Ausgabe eines 400-Token-Prompts ist
zwischen 0.32.0 und 0.32.1 **bit-identisch** (1696 Zeichen). Wichtig ist an
0.32.1 etwas anderes: mlx-vlm 0.6.15 enthält bereits
[#1949](https://github.com/Blaizzy/mlx-vlm/pull/1949) („Fix issues + tests with
mlx 0.32.1", gemerged 34 Minuten *vor* dem 0.6.15-Release) — unter anderem
`mx.clear_streams()` beim Thread-Ende und contiguous Views bei der
KV-Dequantisierung. Der Stand passt also zusammen.

> **Offen: mlx 0.32.2.** Für Qwen3.8 hängt daran der einzige echte
> Kernel-Gewinn: `head_dim 256` bekommt wieder einen fused Full-Attention-Pfad
> ([#4185](https://github.com/ml-explore/mlx/pull/4185), plus
> [#3842](https://github.com/ml-explore/mlx/pull/3842) für NAX/M5). Beide sind
> **nach** dem `0.32.1`-Tag gemerged; auf PyPI gibt es 0.32.2 noch nicht (auch
> nicht als `mlx-metal`/`mlx-cpu`), GitHub-Releases führen keine Wheels, und
> conda-forge steht auf 0.32.0. Ein Quellbau scheitert an
> `xcrun: unable to find utility "metal"` — der Metal-Compiler steckt in
> Xcode.app, die Command Line Tools allein reichen nicht.
> Patch `0013` liegt deshalb fertig im Baum und ist auf 0.32.0 **und** 0.32.1
> als **inert verifiziert** (`_FORCE_FUSED == False`). Er schaltet sich von
> selbst scharf, sobald 0.32.2 installiert ist — der Versionsbump steht bereits
> auf `main`.

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
| `APC_ENTRIES` | 1 | 2 | 3 mit `wired_limit`, sonst 2 |
| `KV_BITS` | 8 (ab 8k Token) | — (f16) | — (f16) |
| `PREFILL_STEP` | 512 | 1024 | 2048 |
| `VISION_CACHE` | 1 | 4 | 20 |
| `APC_DISK_MAX_GB` | 40 | 40 | 80 (plattenabhängig gekappt) |
| **Peak-RAM** | **~18,8 GiB** @ 29k | ~25,0 GiB @ 43k | ~30 GiB @ 90k (ger.) |
| `context_length` | 32768 | 49152 | 98304 |

**`PROFILE=auto`** (Default) entscheidet anhand des Metal-Working-Sets:
≥ 30 GiB → `roomy`, ≥ 24 GiB → `balanced`, sonst `lean`. Der Working-Set wird
dafür aus `iogpu.wired_limit_mb` bzw. dem macOS-Default (2/3 des RAM bei
≤ 36 GB, sonst 3/4) geschätzt — die exakte Zahl aus Metal steht danach im
Banner. `default` bleibt als Alias für `balanced` erhalten.

Ein Profil setzt nur *Defaults* — einzelne Env-Variablen gewinnen weiterhin,
z. B. `PROFILE=roomy APC_ENTRIES=2 …` oder `PROFILE=lean KV_BITS=` (zurück auf
f16).

> **`roomy` stand bis 2026-08-20 auf `APC_ENTRIES=4`.** Die Budgetrechnung ist
> ein Worst Case — alle Snapshots gleichzeitig auf voller Promptlänge — und mit
> vier Einträgen (5 Kopien) waren das bei 44 GiB Working-Set nur ~85k Token,
> also weniger als die empfohlenen 98304. Dass das produktiv trotzdem mit
> 131072 trug, lag daran, dass die Snapshots real nie alle gleichzeitig am
> Anschlag stehen: Glück, keine Garantie. Mit `APC_ENTRIES=2` (3 Kopien) sind
> es ~142k Token, das Profil deckt seine eigene Empfehlung jetzt ab.
>
> **Aber nur mit gesetztem `wired_limit`.** Ohne `sudo sysctl -w
> iogpu.wired_limit_mb=45056` gibt macOS auf 48 GB nur 37,4 GiB Working-Set,
> und daraus werden ~107k Token — 98304 trägt, 131072 nicht. Die maßgebliche
> Zahl steht bei jedem Start in der Zeile `→ KONTEXT-BUDGET` im Banner.
>
> **Seit 2026-08-20 hängt auch `APC_ENTRIES` daran** (s. „Kalte Prefills"). Das
> Skript liest `iogpu.wired_limit_mb` selbst und wählt auf `roomy` drei
> Snapshots statt zwei, sobald das Limit steht:
>
> | | 3 Kopien (`APC_ENTRIES=2`) | 4 Kopien (`APC_ENTRIES=3`) |
> |---|---|---|
> | ohne `wired_limit` (36 GiB WS) | ~103k Token | ~77k — **unter 98304** |
> | mit `wired_limit 45056` (44 GiB WS) | ~147k Token | ~109k Token |
>
> Ohne Limit bleibt es deshalb bei 2. Kalte Prefills gegen
> `[METAL] Insufficient Memory` zu tauschen wäre der schlechtere Handel.

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

### Kalte Prefills sind der teuerste Posten — nicht der Decode

Auswertung des Produktivlogs vom 2026-08-17 bis -20 (18 335 Zeilen):

| | |
|---|---|
| Prefills mit `cached_tokens`-Angabe | 549 |
| davon `cached_tokens=0` | **168 (30,6 %)** |
| davon über 8k Token | 45 |
| deren Prefill-Zeit zusammen | **2415 s** |

Zum Vergleich: der DFlash-2-Gewinn gegenüber MTP sind +17 % auf Antworten von
rund 10 s, also ~1,5 s pro Antwort. **Ein einziger vermiedener 23k-Kaltprefill
(54,9 s gemessen) wiegt rund 36 solche Antworten auf.** Wer hier optimiert,
optimiert an der Trefferquote des Prefix-Cache, nicht am Drafter.

Ein Teil der Misses ist unvermeidbar (neue Konversation) oder selbstgemacht (am
2026-08-20 liefen 31 Serverstarts — Messtag). Der Rest nicht. Im Fenster
10:03–10:22 lief **kein** Neustart, trotzdem:

```
10:03:28  prompt=21667  cached=21651   1,4 s
10:07:26  prompt=23857  cached=23613   3,6 s
10:11:37  prompt=21665  cached=0      50,4 s   ←
10:11:38  prompt=21740  cached=21649   0,8 s
10:12:36  prompt=23118  cached=0      56,8 s   ←
10:15:30  prompt=23200  cached=0      70,7 s   ←
10:16:24  prompt=23280  cached=23184   0,8 s
```

Kalte und warme Turns derselben Größe wechseln sich ab — das ist Verdrängung:
mehr als zwei gleichzeitig aktive Konversationen auf zwei Snapshot-Plätzen.
Dazu kam, dass die Rückfallebene löchrig war: der SSD-Tier stand mit 58 GB
exakt am 60-GB-Deckel und räumte bei praktisch jedem Store
(`APC disk: evicted 6 shard(s); now 56341.8 MB / 64424.5 MB cap`).

Zwei Konsequenzen, beide im Start-Skript:

- **`APC_DISK_MAX_GB` auf `roomy` von 60 auf 80** — und generell gekappt auf
  das, was das Volume mit 25 GB Reserve trägt. Das Skript rechnet das beim
  Start aus und meldet die Kappung.
- **`APC_ENTRIES` auf `roomy` von 2 auf 3**, aber nur mit gesetztem
  `iogpu.wired_limit_mb` (s. Profiltabelle) — ohne Limit passt der dritte
  Snapshot rechnerisch nicht ins Budget.

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

## DFlash 2 ist der Default-Drafter

[DFlash 2](https://inco.ai/blog/dflash2/) läuft über `patches/0040` (Module,
= Upstream-PR [#1959](https://github.com/Blaizzy/mlx-vlm/pull/1959)) und
`patches/0021` (Prefix-Cache-Routing) und ist seit 2026-08-20 der Default.
Zurück auf den MTP-Kopf: `DRAFT_KIND=mtp ./start-mlx_qwen3.8.sh`.

Hintergrund: mlx-vlm implementiert bis 0.6.15 und auf `main` nur DFlash **v1**,
oMLX 0.6.2 ebenso — für Qwen3.8-27B existiert aber ausschließlich ein
v2-Drafter. Bis 2026-08-20 lief das hier über eine eigene Transkription der
MLX-Referenz von z-lab
([`dflash/model_mlx.py`](https://github.com/z-lab/dflash/blob/main/dflash/model_mlx.py),
Patch `0020`), geprüft gegen die Referenz (Conv `max|diff| = 0`, identische
Selector-Pfade) und gegen den Checkpoint (81/81 Parameter in Name und Form).

**Seit 2026-08-20 kommt der Code stattdessen aus Upstream-PR #1959.** Die eigene
Transkription war korrekt — inklusive des Codebook-Renames
(`candidate_selector.{predecessor,successor}_codebook` → `…weight`), den z-lab
selbst erst am 2026-08-18 mit
[`e128a7e`](https://github.com/z-lab/dflash/commit/e128a7e) kanonisierte und den
#1959 identisch macht. Ersetzt wurde sie trotzdem, weil #1959 drei Dinge
mitbringt, die sie nicht hatte:

- einen dedizierten **bit-exakten 4bit-M=4-Metal-Verifier-Kernel**, der die vier
  Verify-Zeilen zusammen streamt und die packed weights über alle vier Token
  wiederverwendet
- **verteilungserhaltendes Rejection Sampling** für `temperature > 0` — die
  eigene Fassung war nur gegen greedy auf Bit-Gleichheit geprüft
- optionale **In-Memory-Quantisierung** des Drafters (`MLX_VLM_DRAFT_BITS`)

Der vorhandene Checkpoint `Qwen3.8-27B-DFlash2-4bit` lädt damit unverändert
(`DFlash2DraftModel`, 179 Parameter, 1,008 GiB) — keine Neukonvertierung nötig.
Upstream misst auf einem M3 Ultra mit BF16-Drafter und `block_size 4`:
31,85 → 47,07 t/s (1,48×) bei 500/500 identischen Token und 60,5 % Acceptance.
Was #1959 **nicht** hat, ist der Guard gegen korrupte Bonus-Tokens — der bleibt
als `0041` lokal.

### Messung: ohne Drafter / MTP / DFlash 2

Identische Prompts, `temperature 0`, Decode-Rate aus dem `predicted_ms` des
Servers, Median aus drei Läufen:

| Fall | ohne Drafter | MTP | DFlash 2 | |
|---|---|---|---|---|
| JSON | 16,4 t/s | 37,8 t/s | **44,2 t/s** | +17 % |
| Code | 16,4 t/s | 35,8 t/s | **42,6 t/s** | +19 % |
| Langkontext (5,8k) | 14,2 t/s | 36,0 t/s | **40,8 t/s** | +13 % |
| Tool-Call | 18,3 t/s | 33,8 t/s | **39,8 t/s** | +18 % |
| Prosa | 16,5 t/s | **29,8 t/s** | 29,9 t/s | ±0 % |

**Der Gewinn steckt in strukturierter Ausgabe** — Tool-Calls, JSON, Code — und
damit genau in der Agentenlast. Bei freier Prosa sind beide gleichauf; dort
liegt MTP bei der Acceptance sogar vorn (57 % gegen 45 %).

Interessant ist, *warum*: die Acceptance-**Rate** ist bei beiden praktisch
identisch (Median 81 % gegen 80 %). DFlash 2 draftet pro Runde schlicht mehr
Token (`block_size 4` statt 3) und gewinnt darüber. Genau deshalb ist die
Blockgröße der empfindlichste Parameter — Sweep gegen MTP: `3` +6 %, **`4` +19 %,
`5` +20 %**, `8` +6 %. Der Checkpoint ist auf `block_size 8` ausgelegt, was auf
einem 4bit-Target die schlechteste Wahl ist; z-lab empfiehlt für quantisierte
MLX-Modelle ebenfalls ≤ 5.

Korrektheit: Ausgabe bei `temperature 0` in allen fünf Fällen identisch zum Lauf
**ohne** Drafter, Tool-Call-Argumente identisch. Beide Drafter erreichen ~2,1×
gegenüber gar keinem Drafter.

Kosten: der Drafter belegt 1,01 GiB statt 0,23 GiB. Auf `lean` und `balanced`
ist das direkt weniger Kontext — dort lohnt die Abwägung, ob `DRAFT_KIND=mtp`
die bessere Wahl ist.

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

### Die verbleibende Patch-Abhängigkeit

DFlash 2 hängt weiter an **zwei** Patches: `0040` für die Drafter-Module und
`0021` fürs Prefix-Cache-Routing. Ein `pip install -U mlx-vlm` ohne
anschließendes `apply-patches.sh` macht den Drafter unladbar. Das Start-Skript
fängt das ab — es prüft beide und fällt notfalls mit Warnung auf MTP zurück.

Für `0040` ist absehbar, dass die Abhängigkeit entfällt: es *ist* der
Upstream-PR. Für `0021` nicht: das zugehörige Issue
[#1966](https://github.com/Blaizzy/mlx-vlm/issues/1966) wurde am 2026-08-20
**geschlossen** — zugunsten von
[#1923](https://github.com/Blaizzy/mlx-vlm/pull/1923) („conservative DFlash APC
prefix reuse", nur `B=1`, text-only, exact-prefix). Der hier gefahrene Ansatz
(Batch-Pfad über `MLX_VLM_SPECULATIVE_BATCH=1`) landet also nicht; bis #1923
gemerged ist, bleibt `0021` lokal.

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

# Ist die Maschine mitten in einem Request eingeschlafen? (s.u.)
pmset -g log | grep -E "Entering Sleep state|Wake Requests" | tail -5
```

> **Auf Akku schläft der Mac mitten in die Generierung hinein.** Signatur im
> Log: `Decode completed` meldet eine plausible `elapsed`- und `rate`-Zahl,
> aber zwischen zwei `Decode progress`-Zeilen springt die **Wanduhr** um
> Minuten. Real gemessen am 2026-08-21: ein 400-Token-Request stand zwischen
> Token 210 und 220 **989,7 s** still, während der Decode-Zähler nur 0,49 s
> zählte. `pmset -g log` zeigte dazu passend
> `06:22:20 Entering Sleep state due to 'Idle Sleep' … Using Batt` und einen
> Weckauftrag mit `deltaSecs=991`.
> Für Messungen und für jeden Agent-Lauf, der länger als der Idle-Timer dauert,
> heißt das: `caffeinate -dimsu` davorsetzen (oder den Server gleich so
> starten). Ohne das misst man Schlafphasen statt Durchsatz.

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

Neun Patches gegen `site-packages`, angewendet von `patches/apply-patches.sh`
(idempotent, `--check` / `--revert`). Sie verschwinden bei jedem
`pip install -U mlx-vlm` — danach erneut ausführen. Die Reihenfolge ist ab
`0040` bindend: `--revert` läuft deshalb rückwärts.

**Eigene:** `0010` (APC-Einzel-Snapshot), `0011` (Rollen-Kompatibilität),
`0012` (Decode-Rate im Log), `0013` (fused Attention für `head_dim` 256),
`0014` (`QUANT_KV_START` auf dem uniform-Pfad), `0021` (Prefix-Cache-Routing
für Nicht-MTP-Drafter), `0041` (Guard gegen korrupte DFlash-Bonus-Tokens).

**Fremde, noch offene Upstream-PRs** — alle selbst reproduziert und
gegengetestet:

| Patch | Wirkung | Betrifft uns |
|---|---|---|
| `0040` = [#1959](https://github.com/Blaizzy/mlx-vlm/pull/1959) | DFlash 2 upstream: exakter 4bit-M=4-Verifier-Kernel, verteilungserhaltendes Rejection Sampling für `temperature > 0`, In-Memory-Drafter-Quantisierung | **ersetzt den eigenen Patch `0020`**; muss als letzter Patch laufen |
| `0030` = [#1956](https://github.com/Blaizzy/mlx-vlm/pull/1956) | `KV_BITS` + Drafter + Batch-Cache stirbt an `AttributeError: 'tuple' object has no attribute 'shape'` | **Normalbetrieb auf `lean`** — s. u. |
| `0031` = [#1835](https://github.com/Blaizzy/mlx-vlm/pull/1835) | Prefix-Wiederverwendung auf nicht-trimmbaren rekurrenten Caches (die 48 GDN-Layer) → `'ArraysCache' object has no attribute 'trim'` | **nicht** über den Server (`_prefix_cache_trim_amount` läuft nur in `stream_generate`); Vorsorge für `chat_ui` und eigene Skripte |

> **Korrektur vom 2026-08-20 zu `0030`:** hier stand, der Patch sei nur bei
> `MAX_NUM_SEQS > 1` relevant. Das galt, solange MTP der Default war. Seit
> DFlash 2 Default ist, setzt das Start-Skript `MLX_VLM_SPECULATIVE_BATCH=1` —
> und `_make_cache` baut den Batch-Cache auch bei `MAX_NUM_SEQS=1`, sobald
> `KV_BITS` gesetzt ist (`generate/ar.py:796`). Auf `PROFILE=lean` ist
> `KV_BITS=8` Default. Dort ist `0030` also Normalbetrieb, nicht Vorsorge.
> Dazu: **#1956 und [#1938](https://github.com/Blaizzy/mlx-vlm/pull/1938) sind
> derselbe Fix von zwei Autoren** — dieselben zwei Dateien, derselbe Inhalt.
> Nur einer wird mergen; `0030` deckt beide ab.

Sobald einer davon upstream gemerged ist, meldet `apply-patches.sh` „KONFLIKT" —
das ist das Signal, die Datei zu löschen.


Im Einzelnen:

- **`0013-force-fused-sdpa-head-dim-256.patch`** — lokal, kein Upstream-PR.
  Qwen3.8 hat `head_dim 256`. mlx' Default-Dispatch lässt fused Full-Attention
  nur für `head_dim` 64/80/128 zu; die 16 Full-Attn-Layer laufen deshalb auf dem
  unfused Graph und materialisieren pro Layer einen Score-Transienten von
  `O(n_heads × qL × kL)` — das ist der eigentliche Grund, warum `PREFILL_STEP`
  hier überhaupt ein RAM-Hebel ist. mlx 0.32.2 ([#4185](https://github.com/ml-explore/mlx/pull/4185))
  stellt die 192/256-Kernel wieder her, erreichbar **nur** über
  `force_fused=True`; der Default-Dispatch routet weiterhin nicht dorthin. Der
  PR begründet das damit, dass nur die Runtime ihr Speicherbudget kennt — was
  hier zutrifft. Eng gefasst: nur `qL > 1` (Prefill/Verify, nicht Decode), nur
  `head_dim` 192/256, ohne Array-Maske, ohne Sinks. **Auf mlx < 0.32.2 inert**
  (Probe beim Import fällt auf `TypeError`; auf 0.32.0 und 0.32.1 verifiziert).
  Rollback: `QWEN38_FORCE_FUSED_SDPA=0`.
- **`0014-quantized-kv-start-uniform.patch`** — lokal, kein Upstream-PR.
  `quantized_kv_start` galt auf dem Batch-Pfad nur für TurboQuant
  (`generate/ar.py:786`, `defer_turbo`). Auf dem uniform-Pfad — `--kv-bits` ohne
  `--kv-quant-scheme turboquant`, also unser Default — wurde **ab Token 0**
  quantisiert, egal was `--quantized-kv-start` sagt. Gemessen mit
  `_make_cache(kv_bits=8, quantized_kv_start=8192)`:

  | `prefill_length` | ohne Patch | mit Patch |
  |---|---|---|
  | 1000 | `BatchQuantizedKVCache` | `BatchKVCache` (f16) |
  | 20000 | `BatchQuantizedKVCache` | `BatchQuantizedKVCache` |

  Betrifft `PROFILE=lean` im Normalbetrieb (dort ist `KV_BITS=8` Default). Die
  Entscheidung fällt wie bei `defer_turbo` **einmal** beim Anlegen des Cache
  anhand der Promptlänge — es wird nicht mitten im Request umgeschaltet.
  Rollback: `QUANT_KV_START=0`.
- **`0041-dflash2-guard-invalid-bonus-token.patch`** — lokal, kein Upstream-PR,
  Nachfolger von `0022`. `propose_block` baut den nächsten Block aus dem
  Bonus-Token des vorigen `_speculative_walk`. Ist der Wert korrupt, wirft
  `mx.array()` nur `RuntimeError: std::bad_cast` — ohne Wert, ohne Index, ohne
  Hinweis, dass es um eine Integer-Konvertierung geht (reproduzierbar mit
  `mx.array([[2**63]], dtype=mx.int32)`). Genau so starb am 2026-08-20 10:07
  ein Request nach 250 Tokens. Der Patch prüft gegen `vocab_size` und nennt den
  Wert. Bewusst kein Clamping: ein still ersetztes Token verfälscht die Ausgabe,
  statt den Bug zu zeigen. **PR #1959 hat diesen Guard nicht** — die Stelle ist
  upstream offen. Der nackte `std::bad_cast` selbst ist ein Upstream-Papercut in
  mlx — `mx.array` sollte bei Integer-Überlauf einen `OverflowError` mit Wert und
  Index werfen, wie es die Nachbarpfade
  (`Invalid type NoneType received in array initialization.`) längst tun.
- **`0012-decode-progress-cumulative-rate.patch`** — lokal, kein Upstream-PR.
  Das `rate=` in `Decode progress` war die Momentanrate zwischen zwei
  Log-Aufrufen (`emitted_tokens / (now - previous_token_at)`). Unter
  spekulativer Dekodierung wird ein akzeptierter Block in Mikrosekunden
  ausgegeben, deshalb meldeten 17 % aller Zeilen über 1000 tok/s (Spitze
  162153) im Wechsel mit viel zu niedrigen Werten — im Widerspruch zum
  `elapsed=` derselben Zeile. `rate=` ist jetzt die kumulative Rate wie in
  `Decode completed`, die Momentanrate bleibt als `inst=`.
- **`0011-role-compat-developer-to-system.patch`** — lokal, kein Upstream-PR.
  Das Qwen3.8-Template kennt nur `system/user/assistant/tool` und wirft bei
  allem anderen `Unexpected message role.` → HTTP 500. Das Request-Schema
  erlaubt aber zusätzlich `developer`; genau diese eine Rolle rutscht durch die
  Validierung in die Template-Ausnahme (Hermes schickt sie). Der Patch bildet
  `developer → system` und `function → tool` ab, alles andere fällt weiterhin
  durch.
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
