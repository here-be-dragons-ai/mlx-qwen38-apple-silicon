# Qwen3.8-27B (MLX 4bit) on Apple Silicon

Setup and start scripts for running **Qwen3.8-27B** as a local, OpenAI-compatible
server (`mlx-vlm`) on an Apple Silicon Mac — including the memory sizing that
decides between "runs usefully" and "dies with
`[METAL] Insufficient Memory`".

The weights (14.95 GiB) are **not** the problem. The bottleneck is the KV cache
at **64 KiB per token**, which is paid per copy (running sequence + every
prefix-cache snapshot). It determines how much context is left — and therefore
whether this thing is usable as an agent backend.

Three profiles cover the usual machines; `PROFILE=auto` (default) picks one:

| profile | target machine | peak RAM | `context_length` |
|---|---|---|---|
| `lean` | 32 GB **without** `sudo` | ~18.8 GiB | 32768 |
| `balanced` | 32 GB with `wired_limit 26624` | ~25.0 GiB | 49152 |
| `roomy` | 48 GB with `wired_limit 40960` | ~33 GiB | 65536 |

Tested on macOS 26. `roomy` is the original M5 Pro setup, run in production for
weeks; the 32 GB profiles are derived from it. The start script determines the
memory budget at runtime from the machine's real values and warns when the
client configuration exceeds it.

> **`roomy` was set to `wired_limit 45056` until 2026-08-21.** That is 44 GiB of
> Metal working set on a 48 GiB machine — macOS is left with ~2 GiB, and that is
> not enough. On 2026-08-21 at 12:30 the test machine therefore threw a **kernel
> panic** (`watchdog timeout: no checkins from watchdogd in 93 seconds`):
> 45.8 GiB of 48 GiB were *wired*, the pageable pool was below 1 MB, and the
> pageout scanner got back only 252 of 3,086 requested pages. The trap:
> `memoryPressure` reported `false`, because macOS' pressure metric evaluates the
> compressor and **not** wired memory — there is no jetsam that steps in first
> for GPU wired memory. The limit is the only protection, hence 40960 now
> (40 GiB, leaving macOS 8 GiB).

---

## Requirements

| | |
|---|---|
| Hardware | Apple Silicon (arm64), ≥ 32 GB unified memory |
| macOS | current, with Xcode Command Line Tools (`xcode-select --install`) |
| Xcode | **only for the mlx source build** (see below): Xcode.app + Metal toolchain. Everything runs without it, only `head_dim 256` stays unfused |
| Disk | ~20 GB for model + drafter, plus up to 40 GB for the SSD prefix cache |
| Network | one-off ~15 GB download from HuggingFace |

Everything else (uv, venv, Python 3.12, mlx-vlm) is installed by
`install-prereqs.sh`.

---

## Installation

```sh
git clone https://github.com/here-be-dragons-ai/mlx-qwen38-apple-silicon.git
cd mlx-qwen38-apple-silicon

# 1. Software + model weights (idempotent, downloads are resumable)
./install-prereqs.sh

# 2. Raise the GPU wired limit — the most important step (see below)
sudo sysctl -w iogpu.wired_limit_mb=26624    # 32 GB;  48 GB -> 40960

# 3. Start the server (127.0.0.1:8888)
./start-mlx_qwen3.8.sh
```

On 32 GB, step 2 is the difference between ~23k and ~48k usable context. The
setup runs without it — `PROFILE=auto` detects that and switches to `lean`.

`install-prereqs.sh` creates or verifies: Xcode CLT → [uv](https://astral.sh/uv)
→ venv under `~/src/mlx/.venv` (Python 3.12) → mlx-vlm + dependencies → Metal
self-test → patches → model + MTP drafter → `~/.mlx-qwen38/{logs,apc}`.

| option | effect |
|---|---|
| `--check` | verify only, change nothing |
| `--skip-model` | software yes, 15 GB download no |
| `--latest` | newest instead of the pinned versions |

Paths are controllable via env: `MLX_HOME` (default `~/src/mlx`), `MLX_MODELS`,
`PYTHON_VERSION`.

**Pinned, verified state:** `mlx 0.32.2.dev` **from the source build** (see
below; `install-prereqs.sh` still pins `mlx 0.32.1` from PyPI), `mlx-lm 0.31.3`,
**`mlx-vlm 0.6.15`**, `transformers 5.15.0`, `numpy 2.5.2`,
`huggingface-hub 1.27.0`, `pillow 12.3.0`, Python 3.12.
`mlx-vlm >= 0.6.13` is mandatory — only there is prefix caching correct upstream
(`semantic_extra_hash()`); before that the cache only hits on byte-identical
prompts. From `0.6.14` onwards it also contains the short-prompt fix (PR #1901),
which previously required a local patch.

`mlx 0.32.1` is a pure compatibility step and **costs nothing**. Measured on
this machine (12 runs per version, `temperature 0`, fixed prompts, decode rate
from the server log):

| | median | mean | min–max |
|---|---|---|---|
| `mlx 0.32.0` | 43.2 t/s | 43.9 t/s | 42.8–45.4 |
| `mlx 0.32.1` | 43.3 t/s | 43.3 t/s | 40.7–47.2 |

Same token count in both runs, and the output of a 400-token prompt is
**bit-identical** between 0.32.0 and 0.32.1 (1,696 characters). What matters
about 0.32.1 is something else: mlx-vlm 0.6.15 already contains
[#1949](https://github.com/Blaizzy/mlx-vlm/pull/1949) ("Fix issues + tests with
mlx 0.32.1", merged 34 minutes *before* the 0.6.15 release) — among other things
`mx.clear_streams()` on thread exit and contiguous views during KV
dequantization. So the versions fit together.

### mlx 0.32.2 — built from source since 2026-08-21

For Qwen3.8 the only real kernel win hangs on this: `head_dim 256` gets a fused
full-attention path again
([#4185](https://github.com/ml-explore/mlx/pull/4185), plus
[#3842](https://github.com/ml-explore/mlx/pull/3842) for NAX/M5). Both were
merged **after** the `0.32.1` tag, and **0.32.2 does not exist on PyPI to this
day** (not as `mlx-metal`/`mlx-cpu` either). That is why mlx is built here:

```sh
# Prerequisite: Xcode.app + Metal toolchain. The Command Line Tools are NOT
# enough — without them: xcrun: unable to find utility "metal"
sudo xcodebuild -license accept
xcodebuild -downloadComponent MetalToolchain      # ~690 MB
xcrun -sdk macosx metal --version                 # must succeed

git clone https://github.com/ml-explore/mlx.git ~/src/mlx-core
cd ~/src/mlx-core
CMAKE_GENERATOR=Ninja CMAKE_BUILD_PARALLEL_LEVEL=14 \
  uv build --wheel --python ~/src/mlx/.venv/bin/python

# mlx and mlx-metal share the mlx/ directory; the source build is monolithic,
# so remove BOTH first
uv pip uninstall --python ~/src/mlx/.venv/bin/python mlx mlx-metal
uv pip install   --python ~/src/mlx/.venv/bin/python dist/mlx-0.32.2.dev*.whl
```

> **`--python` is mandatory.** Without it `uv` builds against its default Python
> and drops a `cp313` wheel; the venv runs 3.12, and the wheel then does not fit.
>
> **`uv pip install -U mlx` falls back to 0.32.1** and disables patch `0013`
> again. The start banner shows it (`Full-Attn: unfused …`), but only after the
> next restart.

Patch `0013` is therefore live (`_FORCE_FUSED == True`). Measured with
`0.32.2.dev20260821+a082cb91`, production shape `qL=512` / `kL=22747`, 24 heads /
4 KV heads / `head_dim 256`:

| | peak |
|---|---|
| `force_fused=False` | 662 MiB |
| `force_fused=True` | **123 MiB** |

The difference of 539 MiB is the score transient (533 MiB in theory). The blowup
depends on the **asymmetric shape `qL << kL`**, not on the mask type — a
benchmark with `qL = kL` does *not* show it. Practical consequence on a 37.4 GiB
working set: the prefill previously died at 22,747 tokens with `[METAL]
Insufficient Memory`; afterwards **37,822 tokens went through in 91.8 s**.

### Making the wired limit persistent

`sysctl -w` does not survive a reboot. Permanently:

```sh
sudo ./install-wired-limit-daemon.sh
```

Until 2026-08-24 this was a chain of four `sudo` lines with continuation
characters — and that **breaks when pasted**: if the line wraps in the wrong
place, zsh executes the target directory as a command
(`permission denied: /usr/local/libexec/`), `launchctl` continues without `sudo`
(`Warning: Expecting a LaunchAgents path …`) and quits with
`Load failed: 5: Input/output error`. Result: directory present, script and
plist not. That `Load failed: 5` had **two** causes at once — missing root *and*
a `ProgramArguments` pointing at a file that had not been installed yet. The
installation script does both in the right order and is idempotent
(`--uninstall` removes it again).

Verify with `sysctl iogpu.wired_limit_mb`; preview without setting:
`sudo /usr/local/libexec/set-iogpu-wired-limit.sh --dry-run`.
`RunAtLoad` makes the daemon take effect immediately — so the value must be
there right after installation, not only after a reboot.

**The value is no longer in the plist.** Until 2026-08-24 it hardcoded `26624` —
the 32 GB value. Installed that way it took 14 GiB of working set away from a
48 GB machine, while `install-prereqs.sh` suggested `45056` next to it: two
different wrong values in one set of instructions, one of them the kernel-panic
value. Now `set-iogpu-wired-limit.sh` computes it at boot from `hw.memsize` —
RAM minus 6 GiB (≤ 32 GB) or 8 GiB (above):

| RAM | `wired_limit_mb` | reserve |
|---|---|---|
| 32 GB | 26624 | 6 GiB |
| 48 GB | **40960** | 8 GiB |
| 64 GB | 57344 | 8 GiB |

The script **clamps upwards, including for values passed by hand** — `45056` on
48 GB becomes `40960`. It also refuses values *below* the macOS default, which
would lower the working set. Different value: a number in
`/etc/iogpu-wired-limit.conf`.

> **The follow-on error was silent.** Without the limit set, `roomy` falls back
> to `PREFILL_STEP=512`, and because the NAX path from PR #3842 requires
> `qL >= 1024`, it no longer engages — the gain from patch 0013 is gone between
> two restarts, without a warning. The start script now reports this case
> explicitly.

---

## Operation

```sh
./start-mlx_qwen3.8.sh                        # PROFILE=auto
PROFILE=lean ./start-mlx_qwen3.8.sh           # minimal RAM, runs without sudo
PROFILE=roomy ./start-mlx_qwen3.8.sh          # 48 GB setup
PORT=8899 ./start-mlx_qwen3.8.sh              # lab instance
ENABLE_SPEC_DECODE=0 ./start-mlx_qwen3.8.sh   # without drafter
```

### Profiles

| | `lean` | `balanced` | `roomy` |
|---|---|---|---|
| target machine | 32 GB without sudo | 32 GB, `wired_limit 26624` | 48 GB, `wired_limit 40960` |
| `APC_ENTRIES` | 1 | 2 | 3 (see measurement below) |
| `KV_BITS` | 8 (from 8k tokens) | — (f16) | — (f16) |
| `PREFILL_STEP` | 512 | 1024 | 2048 (512 without `wired_limit`) |
| `VISION_CACHE` | 1 | 4 | 20 |
| `APC_DISK_MAX_GB` | 40 | 40 | 80 (capped by disk) |
| **peak RAM** | **~18.8 GiB** @ 29k | ~25.0 GiB @ 43k | ~30 GiB @ 90k (est.) |
| `context_length` | 32768 | 49152 | 65536 (see "Why 3") |

> **With the fused path active** (mlx source build, see above) the script raises
> `PREFILL_STEP` to at least **1024** — the NAX kernel from #3842 explicitly
> requires `query_sequence_length >= 1024` and does not engage at 512. The score
> transient, which is why 512 existed in the first place, then disappears. An
> explicit `PREFILL_STEP=…` is left untouched.

**`PROFILE=auto`** (default) decides from the Metal working set:
≥ 30 GiB → `roomy`, ≥ 24 GiB → `balanced`, otherwise `lean`. The working set is
estimated from `iogpu.wired_limit_mb` or the macOS default (2/3 of RAM at
≤ 36 GB, otherwise 3/4) — the exact number from Metal appears in the banner
afterwards. `default` is kept as an alias for `balanced`.

A profile only sets *defaults* — individual env variables still win, e.g.
`PROFILE=roomy APC_ENTRIES=2 …` or `PROFILE=lean KV_BITS=` (back to f16).

> **`roomy` was set to `APC_ENTRIES=4` until 2026-08-20.** The budget
> calculation is a worst case — all snapshots at full prompt length
> simultaneously — and with four entries (5 copies) that was only ~85k tokens at
> a 44 GiB working set, less than the recommended 98304. That it nevertheless
> carried 131072 in production was because the snapshots never all sit at the
> limit at the same time: luck, not a guarantee. With `APC_ENTRIES=3` (4 copies)
> it is ~142k tokens, so the profile now covers its own recommendation.
>
> **But only with `wired_limit` set.** Without `sudo sysctl -w
> iogpu.wired_limit_mb=40960`, macOS gives only 37.4 GiB of working set on
> 48 GB, and that yields ~107k tokens — 98304 carries, 131072 does not. The
> authoritative number appears on every start in the `→ KONTEXT-BUDGET` line of
> the banner.
>
> **The 44 GiB numbers here are historical.** They come from `wired_limit 45056`
> before the kernel panic of 2026-08-21 and from the time before the fused path.
> With patch `0013` active the prefill transient drops out of the budget
> calculation and the budgets are considerably higher — on a 37.4 GiB working
> set the banner reports ~161k tokens. Trust the banner, not these tables.
>
> **From 2026-08-20 to 2026-08-21 `APC_ENTRIES` also hung on `wired_limit`** —
> three snapshots instead of two as soon as the limit was set. That is
> **reverted**. The coupling rested on exactly the budget calculation that the
> memory sampler disproved on 2026-08-21 (see "What the sampler measures"). It
> had an unpleasant side effect: `sudo sysctl -w iogpu.wired_limit_mb=40960`
> raised the ceiling by 2.5 GiB but doubled the snapshot count at the same time —
> a net regression. `roomy` has been fixed at `APC_ENTRIES=3` since 2026-08-24 —
> the measurement that argued for 1 is explained by patch 0015 (see "What the
> sampler measures"), and the 3 holds now that `_CTX_HINT` is 65536 (see
> "Why 3").

On start the script prints the computed memory budget of this machine:

```
  ──────────── Speicherbudget (gerechnet, keine Messung) ──────
  RAM              : 32 GiB
  Metal-Working-Set: 26.0 GiB   (folgt iogpu.wired_limit_mb)
  Gewichte         : 15.2 GiB   (+1,5 GiB Reserve fuer Aktivierungen)
  frei fuer KV     : 8.9 GiB  auf 3 Kopien (1 live + APC)
  →  KONTEXT-BUDGET: ~48486 Token pro Konversation
```

Most important env switches (all documented with rationale in the script
header):

| variable | default | effect |
|---|---|---|
| `PROFILE` | `auto` | `lean` / `balanced` / `roomy` (table above) |
| `KV_BITS` | per profile | `8` halves 64 → 32 KiB/token, doubles the context |
| `APC_ENTRIES` | per profile | prefix-cache snapshots = conversations kept warm |
| `ENABLE_SPEC_DECODE` | `1` | MTP drafter, +58…132% decode |
| `PREFILL_STEP` | per profile | prefill chunk = transient activation peak |
| `BIND_HOST` / `PORT` | `127.0.0.1` / `8888` | bind address |
| `MODEL_ALIAS` | `Qwen3.8-27B-local` | **must** match the model name in the request |

> With `mlx-vlm` the `model` string from the request **is** the load path — there
> is no `--alias` like in llama.cpp. If the name differs, the server discards the
> loaded model and starts a HuggingFace download (→ 401, even though the model is
> local). The script creates a symlink for this and warns on mismatch.

---

## What the sampler measures

Since 2026-08-21 the start script writes a memory line to `server.log` every 5 s
(`MEM_PROBE_INTERVAL=0` disables it; from 85% it becomes a `WARNING`):

```
2026-08-21 14:23:13 - WARNING - mem active=37.78 cache=0.13 sum=37.91 GiB (95% von 40.00 GiB Working-Set) peak=39.49 GiB
```

The sampler does **not** hook into `mlx-vlm` — the script starts
`mlx_vlm.server` through a `runpy` bootstrap with a daemon thread in front. A
patch in `site-packages` would be gone after every `mlx-vlm` update. From
outside, the number is not measurable: RSS does not include the Metal buffers
(measured 2.3 GiB RSS with 16 GiB of weights).

**What it found.** Two items that appeared in no calculation.

**The fixed floor: ~9 GiB, resolved.** `active` jumped from 15.96 to 25.42 GiB
on the first generation — independent of context length (a 16-token request
triggers it just as much as 44,452 tokens) and never released. A probe around
`mx.eval` showed the source:

```
8.71 GiB cumulative  n=128  language.py:1098  _target_verify_quantized_linears
```

`_fused_quantized_linears()` concatenates the QKV and MLP weights of each layer
into a fused tensor and attaches it to the module permanently — a second copy of
the quantized weights. Not a leak: an optimisation nobody tears down again.
Patch `0015` makes it switchable, and the start script switches it off. Result:
`active` after the first request is **16.70 instead of 25.42 GiB**.

**A "creep per request" does not exist.** It was estimated at ~0.6 GiB per
request for a while — that was a measurement error. In the affected series the
**context length** grew with every request, and what looked like retention was
simply the growing KV cache of the running conversation. The counter-test at
**constant** length (8 × 13,460 tokens, APC off) shows the idle value unmoved
from the second request onwards:

| request | idle before | peak | Δ idle |
|---|---|---|---|
| 1 | 15.96 GiB | 18.73 GiB | — |
| 2 | 18.14 GiB | 21.20 GiB | +2.18 |
| 3 | 17.85 GiB | 21.07 GiB | −0.29 |
| 4 | 18.14 GiB | 21.26 GiB | +0.29 |
| 5–8 | **18.14 GiB** | 21.1–21.5 GiB | **0.000** |

The one-off jump is the KV cache of *one* conversation (13,460 × 64 KiB =
0.82 GiB) plus the remaining ~1.3 GiB floor. After that: nothing. Memory follows
context length exactly as the calculation predicts — there is no leak.

**What became possible afterwards.** Single cold requests after patch 0015,
40 GiB working set:

| prompt | result |
|---|---|
| 85,730 tokens | ✅ 264 s, peak 89% |
| 128,570 tokens | ✅ 464 s, peak 95% |
| ~171,000 tokens | ❌ — but at the **600 s timeout**, not on memory |

Before patch 0015 the same machine died in the agent chain at ~23k tokens.

**Why the sampler stays.** The memory calculation below is an upper bound, not a
promise. First run with the sampler, `roomy`, 40 GiB working set,
`APC_ENTRIES=3`, still **without** patch 0015:

| point in time | `active` |
|---|---|
| after model + drafter, idle | **15.96 GiB** ← baseline, matches the table |
| after 1 request of 13,112 tokens | 27.35 GiB ← **+11.4 GiB** |
| after 4 requests | 35.54 GiB |
| at 23,091 tokens | 37.78 GiB → 95%, the next prefill died in OOM |

The calculation budgets 1.6 GiB for that 13k request (4 copies × 32 KiB/token).
In reality it is 11.4 — **factor 7** — and `active` does not fall back between
requests. `cache` stays near zero throughout, so the buffer cache is not the
cause: mlx releases it correctly at `gc_limit_ = 0.95 × working set`
(`mlx/backend/metal/allocator.cpp`). This is **live** memory, and below the
working set there is no brake on it.

**Resolved — it was the floor, not the cache.** This measurement ran *without*
patch 0015. Of the 11.4 GiB, **9.46 GiB** is the fixed floor from
`_fused_quantized_linears()` (see "The fixed floor" above): 11.4 − 9.46 =
**1.94 GiB** against a budgeted 1.2–1.6 — that matches, and the "factor 7" is
gone. That `active` did not fall back between requests had the same cause; the
suspected "creep per request" was additionally a measurement error (growing
context length, counter-test at constant length above).

That removes the justification for `APC_ENTRIES=1`. `roomy` has been at **3**
since 2026-08-24 (initially 2, see "Why 3" below). What the 1 cost, from
`server.log` (810 prefills, 118 server runs, 37.9% cold):

| | count | prefill time |
|---|---|---|
| warm | 503 | 9.71 M tokens reused |
| cold < 8k | 184 | 602 s (irrelevant) |
| cold 8–20k | 40 | 1128 s |
| cold > 20k | 83 | 5331 s |
| **cold ≥ 8k** | **123** | **107.7 min** |

Those 107.7 min are **two** faults, not one:

* **72.7 min** (81 prefills) follow an earlier large request within the same
  server run — eviction with too few snapshot slots (`in_flight=2` appears 5× in
  the log). This is the upper bound: some of them are genuinely new
  conversations that no additional slot would save. Fix: `APC_ENTRIES`.
* **34.9 min** (42 prefills) are the *first* large request of their run — those
  the disk tier should have caught. Fix: namespace GC + corrected cap
  arithmetic (see below), not `APC_ENTRIES`.

#### Why 3 — and why it depends on the clients' `context_length`

This said **2** at first, computed against the then-current `_CTX_HINT` of
98,304. That calculation was not wrong, it answered the wrong question — and the
setting therefore **never took effect**.

**`mlx-vlm` has no `-c` flag.** The server context comes from `config.json`
(262144); the effective ceiling is the *client's* `context_length` alone.
`_CTX_HINT` is a recommendation for the banner and the input to the overbooking
guard — **not a server limit**. That guard reduces `APC_ENTRIES` while
`ctx_hint > budget × 0.8`. Computed with the script's own budget block
(40.0 GiB working set, 16.0 GiB weights, patch 0013 active):

| `_CTX_HINT` | set | `entries_used` | copies | budget |
|---|---|---|---|---|
| 98304 | 2 | **1** | 2 | 182133 |
| 98304 | 3 | **1** | 2 | 182133 |
| 65536 | 2 | 2 | 3 | 120654 |
| **65536** | **3** | **3** | **4** | **89914** |

At 98,304 *every* setting ends up at 1. Since all three pi instances run
`contextWindow` 65536, `_CTX_HINT=98304` described a client that does not exist —
and reserved memory for prompts nobody sends. Both values are therefore
corrected: `_CTX_HINT` 98304 → **65536**, `APC_ENTRIES` 2 → **3**.

Requirement at 65536 (64 KiB/token f16 — `dequantize_for_apc()`, see box — plus
152 MiB GDN per copy, base 16.70 GiB, usable 0.95 × 40 GiB = 38.0 GiB):

| `APC_ENTRIES` | copies | requirement | |
|---|---|---|---|
| 2 | 3 | 29.1 GiB | |
| **3** | **4** | **33.3 GiB** | 4.7 GiB spare — one warm slot per instance |
| 4 | 5 | 37.4 GiB | too tight |

> **The 3 depends on the clients.** Raising one instance to 98304 requires
> pulling `APC_ENTRIES` back with it — otherwise the guard silently caps to 1 and
> *all* three lose their warm slot. The warning for this is in the start banner
> (`APC_ENTRIES N → M gekappt`); it is easy to miss there.

The `KONTEXT-BUDGET` in the banner remains marked as an upper bound:
**the `mem` lines are authoritative, not the calculation.**

> **`KV_BITS=8` does not help here.** Tempting, because without the score
> transient the 64 KiB/token dominate — but `apc_adapters.py:515` calls
> `dequantize_for_apc()` on snapshot store. Only the live cache is quantized, the
> APC snapshots stay f16. Measured: decode 22.9 → 18.7 tok/s (mean of 6 and 8
> requests respectively), `active` climbed to 37.78 GiB unchanged, same OOM.

### Watchdog: a net, not a requirement

Written while the memory floor was still unexplained. Since patch `0015` it is
**no longer necessary** — memory follows context length and no longer fills up
by itself. As a net for unattended runs it does no harm.
`watchdog-mlx_qwen3.8.sh` runs **instead of** the start script and passes all
variables through:

```sh
./watchdog-mlx_qwen3.8.sh
ENABLE_APC=0 WATCHDOG_PCT=85 ./watchdog-mlx_qwen3.8.sh
```

| variable | default | |
|---|---|---|
| `WATCHDOG_PCT` | `90` | threshold in % of the working set |
| `WATCHDOG_STREAK` | `3` | consecutive measurements above the threshold |
| `WATCHDOG_POLL` | `10` | seconds between two checks |
| `WATCHDOG_MAX_WAIT` | `180` | seconds to wait for idle, then hard restart |

It waits for `in_flight=0` before restarting and also detects a server process
that has died. **90 instead of 95:** at 95% a 29,632-token prefill died at 34% on
2026-08-22 — the threshold has to leave room for the running prefill.

A restart returns to the 15.96 GiB baseline. After patch `0015` this only
matters when a single conversation really does get very long — it no longer has
to run against a floor that grows by itself.

---

## The memory calculation

| item | size | source |
|---|---|---|
| weights, 4bit affine (3 shards) | **14.95 GiB** | file size |
| MTP drafter (speculative decoding) | 0.23 GiB | file size |
| activations, Metal heap, Python | ~1.5 GiB | empirical |
| KV cache | **64 KiB / token** | 16 full-attn layers × 4 KV heads × (256+256) × 2 B |
| GDN state (48 linear layers) | 152 MiB, **length-independent** | 48 × 48 v-heads × 128 × 128 × 4 B (float32) |

Qwen3.8-27B is a **hybrid**: only 16 of the 64 layers are full attention (those
pay KV per token), the other 48 are Gated DeltaNet with a constant recurrent
state. And it is **dense** — no MoE, every token reads all ~15 GiB.

The KV item is paid **per copy**: once for the running sequence, once per
prefix-cache snapshot. With `APC_ENTRIES=3` that is 4 copies:

```
budget = (working set − weights − 1.5 GiB − copies × 152 MiB) / (copies × KV_per_token)
need   = 16.7 GiB + copies × (tokens × KV_per_token + 152 MiB)
```

### How much RAM does it actually need?

Profile `default`, ctx 49152:

| state | RAM |
|---|---|
| weights loaded, idle | 15.2 GiB (measured: RSS 15.5 GiB) |
| + activations / Metal heap / Python | ~16.7 GiB |
| operation, 8k prompt | ~18.7 GiB |
| operation, peak prompt 43k | **~25.0 GiB** ← the sizing case |

Profile `lean` with ctx 32768 peaks at **~18.8 GiB**. The absolute floor for this
model is **~17 GiB** (no drafter, no cache, 8k context, KV4) — below that only a
different model or a smaller quant helps.

On top of that, 5–6 GB for macOS itself; that is why 32 GB tops out at a 26 GiB
wired limit.

Metal's `max_recommended_working_set_size` is **2/3 of RAM by default on Macs
≤ 36 GB** → only 21.33 GiB on 32 GB. `iogpu.wired_limit_mb` sets this value
directly (verified: `wired_limit_mb=45056` → working set exactly 44 GiB; the
value is 1:1, `40960` → 40 GiB). **Not above 40960 on a 48 GiB machine** — at
44 GiB macOS is left with ~2 GiB, and that ends in a kernel panic rather than a
Metal error (see above).

| `iogpu.wired_limit_mb` | KV | budget | recommended `context_length` |
|---|---|---|---|
| default (21.3 GiB) | f16 | ~23k tokens | 24576 — tight, works |
| default (21.3 GiB) | `KV_BITS=8` | ~46k tokens | 32768 |
| **26624 (26 GiB)** | **f16** | **~48k tokens** | **49152 ← recommendation** |
| 26624 (26 GiB) | `KV_BITS=8` | ~97k tokens | 65536 (up to 98304) |

**Recommendation: wired limit 26 GiB, KV f16, context 49152.** That involves not
a single unmeasured trade-off.

Why not higher than 26624: macOS needs 5–6 GB itself. Too high a limit trades
"Metal OOM" for a beachball or a kernel panic. Pure inference machine without
browser/IDE: 28672 works.

---

## Client configuration

The server speaks OpenAI chat completions on `http://localhost:8888/v1`. The
same two rules apply to every client: **model name = alias** and
**context ≤ budget**.

| setting | value | why |
|---|---|---|
| model name | `Qwen3.8-27B-local` | **must** match the symlink name — with mlx-vlm the request model name *is* the load path |
| `base_url` | `http://localhost:8888/v1` | |
| `context_length` | `lean` 32768 · `balanced` 49152 · `roomy` 65536 (see below) | ≤ context budget from the start banner |
| `max_tokens` | 8192 | **do not raise**, see below |
| `reasoning_effort` | `low` | only `low\|medium\|xhigh` — anything else → HTTP 500 |
| `enable_thinking` | `true` | |

A client with YAML configuration in the common `model:` format then looks
roughly like this:

```yaml
model:
  default: Qwen3.8-27B-local
  base_url: http://localhost:8888/v1
  api_key: sk-local
  context_length: 65536
  max_tokens: 8192
  supports_vision: true
  extra_body:
    enable_thinking: true
    reasoning_effort: low
compression:
  threshold: 0.85
```

**Why `max_tokens` must not go up:** many clients trigger compaction at
`(context_length − max_tokens) × threshold`. A larger `max_tokens` therefore
moves the trigger down and wastes context.

Calculation for 65536: (65536 − 8192) × 0.85 → compaction at ~48.7k, peak prompt
~57k — below the budget.

> **Why `roomy` says 65536 here and not the 98304 from the profile table:** the
> profile number applied **with** `wired_limit` set (the table below computes
> with the historical 44 GiB, see above). Without the limit macOS gives only
> 37.4 GiB on 48 GB, and then 98304 is right at the edge — the requirement is
> 35.9 GiB, leaving 0.1 GiB:
>
> That applies to the **unfused** path. With the mlx source build the prefill
> transient disappears, and 98304 carries on 37.4 GiB as well.
>
> | `context_length` | need (3 copies) | at 37.4 GiB | with 44 GiB |
> |---|---|---|---|
> | 131072 | 41.9 GiB | **OOM** | +2.1 GiB tight |
> | 98304 | 35.9 GiB | +0.1 GiB tight | +8.1 GiB |
> | **65536** | **29.9 GiB** | **+7.5 GiB** | +14.1 GiB |
>
> Measured: with 98304 and without the wired limit, every fourth request died
> reproducibly with `[METAL] Insufficient Memory` — at prompts of 17k tokens.
> 65536 carries in both cases. To run 98304, set the limit first.

The start script can check this for you:

```sh
CLIENT_CONFIG=~/path/to/config.yaml ./start-mlx_qwen3.8.sh
```

It then warns when `model.context_length` exceeds the computed budget or
`model.default` does not match the alias. Without the variable the check is
inert.

**Thinking:** the Qwen3.8 chat template accepts only `low|medium|xhigh` and
throws an exception (HTTP 500) on any other value — including `none`. With
nothing specified it defaults to `xhigh`, the most expensive level (measured:
1,269 instead of 428 completion tokens). At ~9 t/s that is a two-minute
difference. If the tool error rate rises, go to `medium` — reasoning that is too
shallow can *increase* total latency through failed attempts.

---

## Speed to expect

Dense means every decode step reads ~15 GiB; that is pure memory bandwidth.
Measured on an M5 Pro, scaled to the lower-bandwidth base variant (~153 GB/s):

| | M5 Pro / 48 GB (measured) | M5 base / 32 GB (**estimated**) |
|---|---|---|
| decode raw | 17.5–18.4 t/s | ~8–10 t/s |
| decode with MTP spec-dec (tool calls/JSON) | 26.9–41.5 t/s | ~13–20 t/s |
| prefill | 420–470 t/s | ~180–250 t/s |

A **cold** 30k prefill therefore takes 2–3 minutes. That is why the prefix cache
and the SSD tier are not an optimisation here but a precondition: measured
89,630 ms → 350 ms for a 36k prompt after a server restart (factor 256).

### Cold prefills are the most expensive item — not decode

Analysis of the production log from 2026-08-17 to -20 (18,335 lines):

| | |
|---|---|
| prefills reporting `cached_tokens` | 549 |
| of those `cached_tokens=0` | **168 (30.6%)** |
| of those above 8k tokens | 45 |
| their combined prefill time | **2415 s** |

For comparison: the DFlash-2 gain over MTP is +17% on answers of roughly 10 s,
so ~1.5 s per answer. **A single avoided 23k cold prefill (54.9 s measured)
outweighs about 36 such answers.** Optimising here means optimising the prefix
cache hit rate, not the drafter.

Part of the misses is unavoidable (new conversation) or self-inflicted (on
2026-08-20 there were 31 server starts — a measurement day). The rest is not. In
the window 10:03–10:22 there was **no** restart, and yet:

```
10:03:28  prompt=21667  cached=21651   1.4 s
10:07:26  prompt=23857  cached=23613   3.6 s
10:11:37  prompt=21665  cached=0      50.4 s   <-
10:11:38  prompt=21740  cached=21649   0.8 s
10:12:36  prompt=23118  cached=0      56.8 s   <-
10:15:30  prompt=23200  cached=0      70.7 s   <-
10:16:24  prompt=23280  cached=23184   0.8 s
```

Cold and warm turns of the same size alternate — that is eviction: more than two
simultaneously active conversations on two snapshot slots. On top of that the
fallback layer was leaky: the SSD tier sat at 58 GB, exactly at the 60 GB cap,
and evicted on practically every store
(`APC disk: evicted 6 shard(s); now 56341.8 MB / 64424.5 MB cap`).

Two consequences, both in the start script:

- **`APC_DISK_MAX_GB` on `roomy` from 60 to 80** — and generally capped to what
  the volume carries with 25 GB in reserve. The script computes this at start and
  reports the capping.
- **`APC_ENTRIES` on `roomy` from 2 to 3.** This was briefly coupled to
  `iogpu.wired_limit_mb` being set; that coupling was reverted on 2026-08-21. The
  value has been fixed at 3 since 2026-08-24 (see "Why 3").

#### The cap is per namespace — the arithmetic was not

Found on 2026-08-24 with the volume at 91%:

```
10G  ...apc/Qwen3.8-27B-local_s1-0c1f0816/   <- 21 Aug, old settings hash
53G  ...apc/Qwen3.8-27B-local_s1-e063a74c/   <- active
```

`apc_disk_namespace()` (`apc.py:221`) fingerprints the directory name from model
path, adapter **and KV quantisation**. Every change to `KV_BITS` therefore
creates a new namespace — the `0c1f0816` above is the `KV_BITS=8` hash from the
sampler measurement. **It is never cleaned up:** the store sets
`self.dir = root/<namespace>` (`apc.py:892`), and `_rebuild_index()` only globs
inside it (`apc.py:1113`). Eviction cannot see foreign namespaces.

Two faults follow, both fixed:

- **The cap acts per namespace, the capping calculation summed across all of
  them.** `du -sk "$APC_DISK"` summed the whole directory and thereby treated
  bytes as reclaimable that eviction can never touch — 10 GB too many here (63
  instead of 53), so 125 GB of cap instead of the correct 115. The script now
  counts the active namespace only.
- **Dead namespace ⇒ GC.** On start, inactive namespaces untouched for longer
  than `APC_NS_KEEP_DAYS` (default 3) are removed. The namespace with the newest
  `mtime` counts as active and is **never** collected — a running server writes
  constantly and is therefore always the newest, which makes the rule robust
  against a second instance as well. Clean immediately:
  `APC_NS_KEEP_DAYS=0 ./start-mlx_qwen3.8.sh`.

This is at the same time the fix for the 34.9 min of cold prefills that the disk
tier should have caught (see above): a tier that thrashes at the cap of the
active namespace while dead weight sits next to it is exactly the leaky fallback
layer that makes a miss cost 50–70 s.

### Clear the SSD tier after every context change

**A snapshot is as large as the context it was written for.** If you shrink
`context_length` later, the old large snapshots remain in the SSD tier — and on
restore they blow the working set even though the budget calculation for the new
context works out.

Measured on 2026-08-21: after changing the client context window from 98304 to
65536, **every fourth request** died with
`[METAL] Command buffer execution failed: Insufficient Memory` — at prompt sizes
of only 17,000 tokens and with 89% of system memory free. A server restart did
not help. The cause was 77 GB of snapshots from the era of 98304 and 131072
tokens.

It was isolated with `ENABLE_APC=0`: the same request then went through
immediately. After `rm -rf ~/.mlx-qwen38/apc/*` and a restart with APC, four
requests in a row without a single memory error, prefill back up to 57,000 tok/s.

```sh
# After every change to context_length / contextWindow:
rm -rf ~/.mlx-qwen38/apc/*
```

The price is low: the tier is a cache, it refills by itself. The first prompts
afterwards run cold.

> **Diagnostic order for `[METAL] Insufficient Memory`** when the budget
> calculation should work out: first `ENABLE_APC=0` to isolate, then clear the
> SSD tier, and only then touch `APC_ENTRIES` or `context_length`. A restart
> alone does **not** clear the tier — it lives on disk.

---

## When it gets too tight

The first move is `PROFILE=lean`. What sits behind it, individually and with its
price (savings relative to a 43k peak prompt):

| lever | saves | price |
|---|---|---|
| lower `APC_ENTRIES` (per step) | **−4.2 GiB** @ 65536 | **higher than long assumed.** The assumption was "small, the SSD tier catches it". Measured over 810 prefills, `APC_ENTRIES=1` cost up to **72.7 min** of cold prefill through eviction — the tier only catches it when it is not thrashing at its own cap (see "The cap is per namespace") |
| `KV_BITS=8 QUANT_KV_START=8192` | **−4.2 GiB** | throughput on the MLX path *not* measured; on llama.cpp, KV quantisation at a comparable place cost up to 8× prefill and 1.9× decode → measure A/B |
| `context_length` 49152 → 32768 | −2.6 GiB | shorter runs before compaction |
| `PREFILL_STEP=512` | ~−0.2 GiB | practically none (prefill is compute-bound, not chunk-bound) |
| weights `mxfp4` instead of `4bit` | −0.78 GiB | `mlx-community/Qwen3.8-27B-mxfp4` = 14.17 instead of 14.95 GiB, supported by mlx 0.32; quality not compared |
| drop the drafter | −0.23 GiB | **a bad trade** — costs 58–132% decode |
| `ENABLE_APC=0` | −5.5 GiB | **unacceptable** — every turn pays the full prefill |

Two things you find while looking that do not help:

- **The vision tower is 0.86 GiB and sits unquantized in BF16 inside the model**
  (5.7% of the weights; the 4bit quantisation skipped it). Dead weight for a
  pure text agent — but mlx-vlm 0.6.13 does not evaluate the
  `language_model_only` switch from `config.json` (no hit in the code). So there
  is no switch; only stripping/requantising the weights by hand.
- **There is nothing ready-made below 4bit.** mlx-community lists 4bit, 8bit,
  mxfp4, nvfp4, oQ4/oQ6 and OptiQ for Qwen3.8-27B — all ≥ 14.17 GiB, no 3bit, no
  DWQ. Quantising yourself would work (`mlx_vlm.convert -q --q-bits 3`,
  ~11.5 GiB) but is a quality experiment with an open outcome.

From here on the honest answer is that 27B dense is tight on 32 GB and a smaller
model would be the better choice.

Do **not** use `--max-kv-size` (rotating KV window): the same idea was disproved
with a needle-in-a-haystack test — the needle outside the window was lost and in
one case even hallucinated (8347 instead of 8342). Silent quality loss is worse
than a cleanly small context.

---

## DFlash 2 is the default drafter

[DFlash 2](https://inco.ai/blog/dflash2/) runs via `patches/0040` (modules,
= upstream PR [#1959](https://github.com/Blaizzy/mlx-vlm/pull/1959)) and
`patches/0021` (prefix-cache routing) and has been the default since 2026-08-20.
Back to the MTP head: `DRAFT_KIND=mtp ./start-mlx_qwen3.8.sh`.

Background: up to 0.6.15 and on `main`, mlx-vlm implements only DFlash **v1**,
as does oMLX 0.6.2 — but for Qwen3.8-27B only a v2 drafter exists. Until
2026-08-20 this ran on an own transcription of the z-lab MLX reference
([`dflash/model_mlx.py`](https://github.com/z-lab/dflash/blob/main/dflash/model_mlx.py),
patch `0020`), verified against the reference (conv `max|diff| = 0`, identical
selector paths) and against the checkpoint (81/81 parameters in name and shape).

**Since 2026-08-20 the code comes from upstream PR #1959 instead.** The own
transcription was correct — including the codebook rename
(`candidate_selector.{predecessor,successor}_codebook` → `…weight`), which z-lab
itself only canonicalised on 2026-08-18 with
[`e128a7e`](https://github.com/z-lab/dflash/commit/e128a7e) and which #1959
makes identical. It was replaced anyway, because #1959 brings three things it
did not have:

- a dedicated **bit-exact 4bit M=4 Metal verifier kernel** that streams the four
  verify rows together and reuses the packed weights across all four tokens
- **distribution-preserving rejection sampling** for `temperature > 0` — the own
  version was only checked for bit equality against greedy
- optional **in-memory quantisation** of the drafter (`MLX_VLM_DRAFT_BITS`)

The existing checkpoint `Qwen3.8-27B-DFlash2-4bit` loads unchanged with it
(`DFlash2DraftModel`, 179 parameters, 1.008 GiB) — no reconversion needed.
Upstream measures on an M3 Ultra with a BF16 drafter and `block_size 4`:
31.85 → 47.07 t/s (1.48×) at 500/500 identical tokens and 60.5% acceptance.
What #1959 does **not** have is the guard against corrupt bonus tokens — that
stays local as `0041`.

### Measurement: no drafter / MTP / DFlash 2

Identical prompts, `temperature 0`, decode rate from the server's `predicted_ms`,
median of three runs:

| case | no drafter | MTP | DFlash 2 | |
|---|---|---|---|---|
| JSON | 16.4 t/s | 37.8 t/s | **44.2 t/s** | +17% |
| code | 16.4 t/s | 35.8 t/s | **42.6 t/s** | +19% |
| long context (5.8k) | 14.2 t/s | 36.0 t/s | **40.8 t/s** | +13% |
| tool call | 18.3 t/s | 33.8 t/s | **39.8 t/s** | +18% |
| prose | 16.5 t/s | **29.8 t/s** | 29.9 t/s | ±0% |

**The gain sits in structured output** — tool calls, JSON, code — and therefore
exactly in the agent workload. On free prose the two are level; there MTP is even
ahead on acceptance (57% against 45%).

What is interesting is *why*: the acceptance **rate** is practically identical
for both (median 81% against 80%). DFlash 2 simply drafts more tokens per round
(`block_size 4` instead of 3) and wins through that. Which is exactly why block
size is the most sensitive parameter — sweep against MTP: `3` +6%, **`4` +19%,
`5` +20%**, `8` +6%. The checkpoint is designed for `block_size 8`, which is the
worst choice on a 4bit target; z-lab likewise recommends ≤ 5 for quantized MLX
models.

Correctness: output at `temperature 0` identical to the run **without** drafter
in all five cases, tool-call arguments identical. Both drafters reach ~2.1× over
no drafter at all.

Cost: the drafter occupies 1.01 GiB instead of 0.23 GiB. On `lean` and
`balanced` that is directly less context — there it is worth weighing whether
`DRAFT_KIND=mtp` is the better choice.

### The prefix cache now works under DFlash2 as well

Originally **every** request under `DRAFT_KIND=dflash` reported
`cached_tokens=0`. Cause found: `server/generation.py` routes every non-MTP
drafter into a second generation loop (`_run_speculative`) that builds its own
prompt cache and never wires up the APC manager. The continuous-batching path
has long been able to do dflash — it is generic over `draft_kind` throughout and
receives `apc_manager`, `draft_kind` and `draft_block_size` on the same line.
Only the switch kept dflash away from it.

`patches/0021-speculative-apc-routing.patch` makes the batch path reachable via
`MLX_VLM_SPECULATIVE_BATCH=1`; the start script sets the variable automatically
as soon as `DRAFT_KIND != mtp`. Measured (5.8k conversation, turn 2):

| | `cached_tokens` | decode 64/66/76 tok | 5767 tok |
|---|---|---|---|
| MTP | 5772 / 5788 | 33.9 / 33.9 / 36.7 t/s | 33.9 t/s |
| DFlash2, old loop | **0** | 40.8 / 38.2 / 45.6 t/s | 38.4 t/s |
| DFlash2, batch path | **5748 / 5788** | 38.7 / 40.7 / 43.0 t/s | **40.9 t/s** |

Throughput therefore stays the same (40.8 instead of 41.5 t/s on average —
noise) and even improves on the long prompt, but the prefix cache is back.
`--draft-block-size` still takes effect (block 4 beats block 8 on both paths),
two parallel requests with `MAX_NUM_SEQS=2` run cleanly, and MTP is unchanged
(`cached=5772`).

### The remaining patch dependency

DFlash 2 still hangs on **two** patches: `0040` for the drafter modules and
`0021` for prefix-cache routing. A `pip install -U mlx-vlm` without a subsequent
`apply-patches.sh` makes the drafter unloadable. The start script catches this —
it checks both and falls back to MTP with a warning if necessary.

For `0040` the dependency will foreseeably disappear: it *is* the upstream PR.
For `0021` it will not: the corresponding issue
[#1966](https://github.com/Blaizzy/mlx-vlm/issues/1966) was **closed** on
2026-08-20 in favour of
[#1923](https://github.com/Blaizzy/mlx-vlm/pull/1923) ("conservative DFlash APC
prefix reuse", `B=1` only, text-only, exact-prefix). The approach used here
(batch path via `MLX_VLM_SPECULATIVE_BATCH=1`) will therefore not land; until
#1923 is merged, `0021` stays local.

---

## Diagnostics

```sh
# Is the prefix cache hitting? (turn 2 must show cached_tokens > 0)
grep -o 'cached[_ ]tokens[=:] *[0-9]*' ~/.mlx-qwen38/logs/server.log | tail -20

# Draft acceptance (below ~40% speculative decoding is not worth it)
grep -i "accept" ~/.mlx-qwen38/logs/server.log | tail -10

# Memory situation
~/src/mlx/.venv/bin/python -c "import mlx.core as mx;print(mx.device_info())"
sysctl iogpu.wired_limit_mb
ps -o rss=,command= -p "$(pgrep -f mlx_vlm.server)" | awk '{printf "%.1f GiB\n", $1/1048576}'

# Patch status (patches live in site-packages and vanish on EVERY pip install)
./patches/apply-patches.sh --check

# Did the machine fall asleep mid-request? (see below)
pmset -g log | grep -E "Entering Sleep state|Wake Requests" | tail -5
```

> **On battery the Mac falls asleep in the middle of generation.** Signature in
> the log: `Decode completed` reports a plausible `elapsed` and `rate`, but the
> **wall clock** jumps by minutes between two `Decode progress` lines. Measured
> on 2026-08-21: a 400-token request stood still for **989.7 s** between tokens
> 210 and 220, while the decode counter recorded only 0.49 s. `pmset -g log`
> showed a matching
> `06:22:20 Entering Sleep state due to 'Idle Sleep' … Using Batt` and a wake
> request with `deltaSecs=991`.
> For measurements and for every agent run longer than the idle timer this means:
> put `caffeinate -dimsu` in front (or start the server that way). Without it you
> measure sleep phases instead of throughput.

| symptom | cause |
|---|---|
| `cached_tokens=1` on large prompts | patch 0002 missing |
| `cached_tokens=0` in turn 2 | mlx-vlm < 0.6.13, or snapshot evicted (`APC_ENTRIES`, patch 0010) |
| `cached_tokens=1` on large prompts | mlx-vlm < 0.6.14 (short-prompt bug, PR #1901) |
| HTTP 401 / HF download on request | model name ≠ alias symlink |
| HTTP 500 on every request | `reasoning_effort` outside `low\|medium\|xhigh` |
| `[METAL] Insufficient Memory` | context over budget → lower `context_length` or `KV_BITS=8` |

---

## Patches

Eleven patches against `site-packages`, applied by `patches/apply-patches.sh`
(idempotent, `--check` / `--revert`). They vanish on every
`pip install -U mlx-vlm` — run it again afterwards. The order is binding from
`0040` onwards, which is why `--revert` runs backwards.

**Own:** `0010` (APC single snapshot), `0011` (role compatibility),
`0012` (decode rate in the log), `0013` (fused attention for `head_dim` 256),
`0014` (`QUANT_KV_START` on the uniform path), `0015` (optional fused quantized
linears), `0021` (prefix-cache routing for non-MTP drafters), `0041` (guard
against corrupt DFlash bonus tokens).

**Foreign, still-open upstream PRs** — all reproduced and cross-tested here:

| patch | effect | affects us |
|---|---|---|
| `0040` = [#1959](https://github.com/Blaizzy/mlx-vlm/pull/1959) | DFlash 2 upstream: exact 4bit M=4 verifier kernel, distribution-preserving rejection sampling for `temperature > 0`, in-memory drafter quantisation | **replaces our own patch `0020`**; must run as the last patch |
| `0030` = [#1956](https://github.com/Blaizzy/mlx-vlm/pull/1956) | `KV_BITS` + drafter + batch cache dies with `AttributeError: 'tuple' object has no attribute 'shape'` | **normal operation on `lean`** — see below |
| `0031` = [#1835](https://github.com/Blaizzy/mlx-vlm/pull/1835) | prefix reuse on non-trimmable recurrent caches (the 48 GDN layers) → `'ArraysCache' object has no attribute 'trim'` | **not** via the server (`_prefix_cache_trim_amount` only runs in `stream_generate`); precaution for `chat_ui` and own scripts |

> **Correction from 2026-08-20 regarding `0030`:** this used to say the patch was
> only relevant at `MAX_NUM_SEQS > 1`. That held while MTP was the default. Since
> DFlash 2 became the default, the start script sets
> `MLX_VLM_SPECULATIVE_BATCH=1` — and `_make_cache` builds the batch cache even
> at `MAX_NUM_SEQS=1` as soon as `KV_BITS` is set (`generate/ar.py:796`). On
> `PROFILE=lean`, `KV_BITS=8` is the default. So `0030` is normal operation
> there, not a precaution.
> Also: **#1956 and [#1938](https://github.com/Blaizzy/mlx-vlm/pull/1938) are the
> same fix by two authors** — same two files, same content. Only one will merge;
> `0030` covers both.

As soon as one of them is merged upstream, `apply-patches.sh` reports
"KONFLIKT" — that is the signal to delete the file.

In detail:

- **`0013-force-fused-sdpa-head-dim-256.patch`** — local, no upstream PR.
  Qwen3.8 has `head_dim 256`. mlx's default dispatch only permits fused full
  attention for `head_dim` 64/80/128; the 16 full-attn layers therefore run on
  the unfused graph and materialise a score transient of `O(n_heads × qL × kL)`
  per layer — which is the actual reason `PREFILL_STEP` is a RAM lever here at
  all. mlx 0.32.2 ([#4185](https://github.com/ml-explore/mlx/pull/4185)) restores
  the 192/256 kernels, reachable **only** via `force_fused=True`; the default
  dispatch still does not route there. The PR justifies this by saying only the
  runtime knows its memory budget — which applies here. Narrowly scoped: only
  `qL > 1` (prefill/verify, not decode), only `head_dim` 192/256, no array mask,
  no sinks. **Inert on mlx < 0.32.2** (the import probe falls to `TypeError`;
  verified on 0.32.0 and 0.32.1). **Active since the source build of 2026-08-21**
  (`_FORCE_FUSED == True`), measured at `qL=512` / `kL=22747`: peak 662 →
  123 MiB, prefill ceiling 22,747 → 37,822 tokens. The start banner shows the
  state in the `Full-Attn:` line. Rollback: `QWEN38_FORCE_FUSED_SDPA=0`.
- **`0015-optional-fused-quantized-linears.patch`** — local, no upstream PR.
  `_fused_quantized_linears()` concatenates the QKV and MLP weights of each layer
  into a fused tensor and attaches it to the module as
  `_qwen3_5_fused_decode_linears` **permanently** — a second copy of the
  quantized weights. This was the fixed memory floor: it appears on the first
  generation, is length-independent and is never released. Measured, idle after
  5 requests on a 40 GiB working set:

  | | with fusion | without fusion | decode (mean of 5 each) |
  |---|---|---|---|
  | with spec decode | 26.00 GiB | **17.00 GiB** | 26.1 vs 25.7 tok/s |
  | without spec decode | 17.08 GiB | **14.96 GiB** | 18.4 vs 18.2 tok/s |

  9 GiB against 1.5%, and the spread of both decode series overlaps completely.
  The patch itself changes nothing — it only adds the switch, the default stays
  upstream behaviour. The fusion is switched off by the start script. Rollback:
  `QWEN38_FUSED_LINEARS=1`.
- **`0014-quantized-kv-start-uniform.patch`** — local, no upstream PR.
  `quantized_kv_start` applied on the batch path only for TurboQuant
  (`generate/ar.py:786`, `defer_turbo`). On the uniform path — `--kv-bits`
  without `--kv-quant-scheme turboquant`, i.e. our default — quantisation
  happened **from token 0**, regardless of what `--quantized-kv-start` said.
  Measured with `_make_cache(kv_bits=8, quantized_kv_start=8192)`:

  | `prefill_length` | without patch | with patch |
  |---|---|---|
  | 1000 | `BatchQuantizedKVCache` | `BatchKVCache` (f16) |
  | 20000 | `BatchQuantizedKVCache` | `BatchQuantizedKVCache` |

  Affects `PROFILE=lean` in normal operation (where `KV_BITS=8` is the default).
  As with `defer_turbo`, the decision is made **once** when the cache is created,
  based on prompt length — there is no switching mid-request. Rollback:
  `QUANT_KV_START=0`.
- **`0041-dflash2-guard-invalid-bonus-token.patch`** — local, no upstream PR,
  successor to `0022`. `propose_block` builds the next block from the bonus token
  of the previous `_speculative_walk`. If the value is corrupt, `mx.array()`
  throws only `RuntimeError: std::bad_cast` — without the value, without an
  index, without any hint that an integer conversion is involved (reproducible
  with `mx.array([[2**63]], dtype=mx.int32)`). That is exactly how a request died
  after 250 tokens on 2026-08-20 at 10:07. The patch checks against `vocab_size`
  and names the value. Deliberately no clamping: a silently replaced token
  corrupts the output instead of showing the bug. **PR #1959 does not have this
  guard** — the spot is open upstream. The bare `std::bad_cast` itself is an
  upstream papercut in mlx — `mx.array` should throw an `OverflowError` with
  value and index on integer overflow, as the neighbouring paths
  (`Invalid type NoneType received in array initialization.`) have long done.
- **`0012-decode-progress-cumulative-rate.patch`** — local, no upstream PR.
  The `rate=` in `Decode progress` was the instantaneous rate between two log
  calls (`emitted_tokens / (now - previous_token_at)`). Under speculative
  decoding an accepted block is emitted in microseconds, so 17% of all lines
  reported over 1000 tok/s (peak 162153) alternating with far too low values — in
  contradiction to the `elapsed=` on the same line. `rate=` is now the cumulative
  rate as in `Decode completed`; the instantaneous rate remains as `inst=`.
- **`0011-role-compat-developer-to-system.patch`** — local, no upstream PR.
  The Qwen3.8 template knows only `system/user/assistant/tool` and throws
  `Unexpected message role.` → HTTP 500 on anything else. The request schema,
  however, additionally allows `developer`; that one role slips through
  validation into the template exception (some clients send it). The patch maps
  `developer → system` and `function → tool`; everything else still falls
  through.
- **`0010-qwen38-apc-single-snapshot.patch`** — local, no upstream PR.
  mlx-vlm stores two nearly identical snapshots per request (checkpoint at
  `len-16` and the full prompt); measurements show the checkpoint is always the
  one that hits. The patch suppresses the second and thereby halves the cache
  memory. Inert without the env variable `QWEN38_APC_SINGLE_SNAPSHOT=1` — that is
  the rollback path.

---

## Files

| file | purpose |
|---|---|
| `install-prereqs.sh` | complete setup from a fresh macOS, idempotent |
| `LICENSE` | MIT No Attribution (SPDX `MIT-0`) |
| `start-mlx_qwen3.8.sh` | server start, profiles lean/balanced/roomy, live budget calculation |
| `watchdog-mlx_qwen3.8.sh` | starts the server and restarts it before memory fills up |
| `download-mlx-model.sh` | resumable HuggingFace downloader (curl, with size check) |
| `patches/apply-patches.sh` | apply / check / revert patches |
| `com.local.iogpu-wired-limit.plist` | LaunchDaemon for the wired limit (calls the script below) |
| `install-wired-limit-daemon.sh` | installs helper + LaunchDaemon in one call, idempotent |
| `set-iogpu-wired-limit.sh` | computes `iogpu.wired_limit_mb` from `hw.memsize`, clamps upwards |

---

## Where the numbers come from

All values marked "measured" come from an **M5 Pro / 48 GB** machine (mlx-vlm
0.6.13/0.6.15, Qwen3.8-27B-4bit, `temperature=0`). Only the hardware-independent
findings are carried over:

- MTP speculative decoding is worth it (decode +58…132%, acceptance 42% prose /
  90% JSON / 93% tool call, quality 7/7 bit-identical)
- prefix caching + SSD tier are worth it (factor 256 on a cold 36k prefill)
- KV windowing is disproved (hallucination outside the window)

**The memory calculation is arithmetic** from `config.json` and the file sizes —
that holds on any machine. **The throughput figures for 32 GB are estimates**,
scaled via memory bandwidth, and marked as such.

---

## License

[MIT No Attribution](LICENSE) (SPDX: `MIT-0`) — MIT without the obligation to
pass on the copyright notice. Copy, adapt and reuse without attribution.
