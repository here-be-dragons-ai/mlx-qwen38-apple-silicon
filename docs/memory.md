# Memory: sizing, measurements, and what was ruled out

This is the long form. The README carries only the numbers you need to run the
thing; everything here is the evidence behind them.

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

---

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

#### Verified end to end (2026-08-25)

The arithmetic above says three conversations stay warm. Measured on mlx-vlm
0.6.16 with `APC_ENTRIES=3`, three distinct conversations of ~15k tokens each,
built in sequence and then continued in the same order:

| | turn 1 (cold) | turn 2 |
|---|---|---|
| A | 14,958 tok, 32.99 s | 14,942 of 14,983 cached, **0.48 s** |
| B | 14,980 tok, 34.57 s | 14,964 of 15,011 cached, **0.49 s** |
| C | 14,988 tok, 33.46 s | 14,972 of 15,019 cached, **0.50 s** |

Three out of three warm, a factor of about 68 on the prefill. Peak memory during
the run was 26.10 GiB, 65% of the working set.

> A first attempt at this test used ~200-token prompts and showed all three cold.
> That was the test being wrong, not the cache: at that size the APC machinery
> (block size 16, checkpoint at len-16) stores nothing worth reusing. Any APC
> measurement needs prompts of a realistic size, or it measures nothing.

> **The 3 depends on the clients.** Raising one instance to 98304 requires
> pulling `APC_ENTRIES` back with it — otherwise the guard silently caps to 1 and
> *all* three lose their warm slot. The warning for this is in the start banner
> (`APC_ENTRIES N -> capped to M`); it is easy to miss there.

The `CONTEXT BUDGET` in the banner remains marked as an upper bound:
**the `mem` lines are authoritative, not the calculation.**

> **`KV_BITS=8` does not help here.** Tempting, because without the score
> transient the 64 KiB/token dominate — but `apc_adapters.py:515` calls
> `dequantize_for_apc()` on snapshot store. Only the live cache is quantized, the
> APC snapshots stay f16. Measured: decode 22.9 → 18.7 tok/s (mean of 6 and 8
> requests respectively), `active` climbed to 37.78 GiB unchanged, same OOM.

---

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

---

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

---

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
