# Qwen3.8-Flash-Next on 48 GB

Flash-Next is a 177B model: 125B body with 6B activated, plus a 51B n-gram PLE
table and a 4B MTP head. 48 layers, 512 experts with 10 active, 12 of the 48
layers are full attention (QSA) and 36 are GatedDeltaNet.

**It runs on this machine and answers correctly. It is not a daily driver: about
4 tok/s in steady state, against 32.84 tok/s for the 27B.** Everything below is
measured on an M5 Pro / 48 GB with `iogpu.wired_limit_mb=40960`, mlx-vlm
0.7.0rc0, mlx 0.32.2, on 2026-09-02.

---

## First: check the checkpoint, not the code

Most MLX conversions of this model in circulation are **broken**, and the failure
is silent at load time and total at generation time — deterministic token salad.

`Qwen4ExpRMSNorm` (`models/qwen4_exp/language.py:576-597`) initialises
`mx.zeros(dim)` and computes `y * (1.0 + weight)`. It therefore requires
**zero-centered** norm weights. Several converters baked the `+1` in at
conversion time, so the runtime shifts a second time. At
`attn_hyper_connection.hc_norm`, whose values span roughly −5…+8, every channel
with `w ∈ (−1, 0)` then flips sign — twice per layer, across all 48 layers, in
the module that mixes the hyper-connection residual streams.

We spent real time blaming the runtime for this. Don't repeat that: the check
costs seconds and needs no download and no GPU. Read the norm tensors and
compare **the same tensor** against a known-good reference.

```python
import json, mlx.core as mx
D = "path/to/checkpoint"
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
REF = {  # (correct, "+1 baked in")
  "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight": (-0.064, +0.936),
  "language_model.model.layers.3.self_attn.q_norm.weight":               (+0.283, +1.283),
  "language_model.model.layers.3.self_attn.indexer.q_layernorm.weight":  (-0.037, +0.963),
  "language_model.model.layers.11.self_attn.k_norm.weight":              (+0.360, +1.360),
}
cache = {}
for k, (good, bad) in REF.items():
    sh = idx[k]
    cache.setdefault(sh, mx.load(f"{D}/{sh}"))
    m = mx.mean(cache[sh][k].astype(mx.float32)).item()
    print(f"{m:+.3f}  {'ok' if abs(m-good) < abs(m-bad) else 'SHIFTED'}  {k}")
```

For a remote repo the same works over HTTP range requests against the
safetensors header — a few KB instead of 100 GB.

> **Do not threshold on the raw mean.** `hc_norm` tensors legitimately sit as
> high as +3.75 while being correctly zero-centered. A first version of this
> check compared means against a fixed cutoff and declared *every* candidate
> broken, including the good one. Only a same-tensor comparison carries.

Surveyed 2026-09-02:

| checkpoint | size | norm |
|---|---|---|
| `mlx-community/Qwen3.8-Flash-Next-4bit` | 111.5 GB | **correct** |
| `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` | 73.5 GB | **correct** |
| `ddalcu/…-Serve-mixed-4-8bit` | 107.3 GB | +1 baked in |
| `Sawfwair/…-Mixed-2bit` | 73.1 GB | +1 baked in |
| `Vontra/…-MLX-oQ2` | 68 GB | +1 baked in, **and** `lm_head` at 2 bit; producer's own card carries a quality hold |

Upstream issue for the whole story: Blaizzy/mlx-vlm#2041.

---

## Where the weight sits

Read from all 22 shard headers of the `mlx-community` 4-bit checkpoint and
summed by category:

| category | GiB | share | streamable |
|---|---:|---:|---|
| routed experts | 70.31 | 67.7 % | yes, `mlx_vlm.moe_offload` |
| PLE / n-gram | 29.80 | 28.7 % | yes, `qwen4_exp/ple_storage.py` |
| rest (attention, norms, hyper-connections) | 2.87 | 2.8 % | resident |
| `embed_tokens` + `lm_head` | 0.74 | 0.8 % | resident |
| shared expert | 0.14 | 0.1 % | resident |
| **total** | **103.86** | | |

**96 % of the model is streamable and the resident core is 3.75 GiB** — measured
after load: 3.76 GiB. That is what makes this feasible at all.

The KV cost is also *lower* than the 27B's, because only 12 of 48 layers are
full attention and they carry 2 KV heads at `head_dim` 256:

| | KV per token | at 65536 ctx |
|---|---|---|
| Flash-Next | **24.0 KiB** | 1.50 GiB |
| Qwen3.8-27B | 64.0 KiB | 9.60 GiB |

---

## Build the runnable view: PLE first, then repack

**Order matters and the wrong order fails late.** The MoE repack rewrites the
checkpoint into `resident-*.safetensors` plus `experts/layer_NNNN.safetensors`,
but leaves `model.safetensors.index.json` pointing at the original shard names
(the expert layout lives in `offload_index.json` instead). `ple_storage`
resolves tensors through that standard index, so running it on a repacked
checkpoint dies with

```
FileNotFoundError: .../Qwen3.8-Flash-Next-4bit-offload/model-00001-of-00022.safetensors
```

after having already written the manifest. Do PLE externalisation on the
original, then repack the view.

### 1. Externalise the PLE table

```python
from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model
prepare_external_ple_model(
    "~/src/mlx/models/Qwen3.8-Flash-Next-4bit",
    "~/src/mlx/models/Qwen3.8-Flash-Next-4bit-ple",
)
```

Hard-links every non-PLE shard and writes `ple-store.json`, so this costs
**0 GB**. 384 PLE tensors leave the parameter tree (3767 → 3383), and 5 of the
22 shards drop out entirely because they held nothing else. The manifest records
`source_root` as a **relative** path (`../Qwen3.8-Flash-Next-4bit`), so the
original checkpoint must stay on disk and must stay on the same filesystem.

`config.json` gains `ple_storage: {"manifest": "ple-store.json", "cache_rows": 0}`,
which `language.py:1037` picks up to build a `QuantizedMMapNGramEmbedding`.

### 2. Repack for expert offloading

```sh
python -m mlx_vlm.moe_offload \
  --build ~/src/mlx/models/Qwen3.8-Flash-Next-4bit-ple \
  --out   ~/src/mlx/models/Qwen3.8-Flash-Next-4bit-final
```

Writes 74 GiB — 104 minus the externalised PLE — in under two minutes on SSD.

> `_check_disk_headroom` (`moe_offload.py:71`) refuses unless free space covers
> **2×** the source size. Its own rationale is that source and output coexist on
> disk, but the source is already there and is not part of free space, so the
> real requirement is ~1× the output. We ran it with `margin=1.2` via a
> monkeypatch rather than disabling it — a mid-write `ENOSPC` truncates output
> silently rather than raising, which is why the guard exists.

Disk after all three directories: 104 + 0 + 74 = 178 GiB.

### 3. Run

```sh
python -m mlx_vlm.generate \
  --model ~/src/mlx/models/Qwen3.8-Flash-Next-4bit-final \
  --expert-cache-gb 34 --prompt "..." --max-tokens 200 --temperature 0.0
```

Load takes 1.1 s (mmap, lazy). Output is correct: *"The capital of France is
Paris."*, `1, 2, 3, 4, 5`, *"Das Haus ist rot."*, and 17 × 24 worked through
with the distributive property.

---

## Throughput: bounded by expert cache hit rate

Steady state, 150 generated tokens, `temperature 0`, varying prompts, median of
the runs after the cache saturates:

| `--expert-cache-gb` | `active` | steady state |
|---:|---:|---:|
| 24 | 26.8 GiB | 2.38 tok/s |
| 26 | 28.6 GiB | 2.61 tok/s |
| 32 | 34.2 GiB | 3.75 tok/s |
| **34** | **36.4 GiB** | **4.17 tok/s** |

Monotonic in cache size, so give it everything the working set allows. 36.4 GiB
peak against a 40 GiB working set is the ceiling here; `wired_limit` stays at
40960 (45056 panics the kernel — see the note in the README).

**The fast numbers are a transient, not a result.** While the cache is still
filling and has not yet had to evict, the same prompt runs at 10–18 tok/s. Once
saturated it settles at 4. A first reading of this looked like a memory-pressure
cliff at `active` ≈ 36.1 GiB; `--expert-cache-gb 26` refutes that, because there
`active` pins at 28.6 GiB and is *slower*. It is simply the hit rate.

And that bound is structural: **70.31 GiB of experts against a ~34 GiB cache is
48 % resident**, with 10 of 512 experts active per layer per token. The misses
come off SSD on every token. For reference, PR #2045 reports 19.04 tok/s for
this model on an M2 Ultra / 128 GB — with everything resident.

### Expert pruning: rejected, for now

`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` prunes to 288 of 512 experts,
which puts the expert set at roughly 39.5 GiB — **~86 % inside the same cache
instead of 48 %**. That would change the miss rate in kind rather than by
degree. It passes the norm check and carries no quality hold.

**Decided against it on 2026-09-02: there are no published benchmarks.** Neither
throughput nor, more importantly, quality. REAP discards 44 % of the experts and
nobody has compared the result against the unpruned model. Buying an unmeasured
speedup with an unmeasured quality regression leaves two unknowns instead of one
problem, and a faster wrong answer is not an improvement.

Revisit if a quality comparison against the unpruned model appears. Until then
the position is that Flash-Next has no usable configuration on 48 GB — not that
one was found.

---

## Verdict

Feasible and correct, not useful. The interesting result is how cheap the
*resident* part is: 3.76 GiB of core and 24 KiB/token of KV for a 177B model.
Everything expensive is streamable, and what is missing is bandwidth, not
memory — 70 GiB of experts against 34 GiB of cache, with the misses on the SSD
path of every token.
