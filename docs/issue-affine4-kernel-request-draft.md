<!-- posted 2026-09-10 as https://github.com/Blaizzy/mlx-vlm/issues/2215
     author: gtonic (the account authenticated in `gh`)
     The H1 below became the issue title and is not part of the posted body.
     Filed on the invitation of @Lazarus-931 when closing #2172 on 2026-09-10:
     "I think an issue/request for specific kernel support in mlx-vlm could be
      a great start. We can circle back and address this."
     Line numbers cite the installed mlx-vlm 0.7.0 in this venv. Nothing below
     cites a locally patched file as upstream code. -->

# Kernel support request: which shapes a packed 4-bit KV path has to accept to be reachable from the server

Follow-up to #2172, which was closed with the suggestion to file exactly this. The request is not for `affine4` specifically and not for a Qwen kernel — it is for a packed-KV backend behind the existing quantized-cache interface, which is what #2172's author proposed in the closing exchange. What follows is the shape list that decides whether such a backend is reachable at all on a single-stream server, and the measurement that says why it matters on this hardware.

## Setup

- Apple M5 Pro, 48 GB, macOS 26, `iogpu.wired_limit_mb=40960` (40 GiB working set)
- mlx-vlm 0.7.0, mlx + mlx-metal 0.32.2, mlx-lm 0.31.3, Python 3.12
- `mlx-community/Qwen3.8-27B-4bit` (`qwen3_5`, hybrid GDN + full attention, 64 layers, **16 of them full-attention**), 24 query heads / 4 KV heads, `head_dim 256`, GQA factor 6
- Server, continuous batching, **one** sequence, exact APC with a disk tier
- Drafter: DFlash 2 (`draft_kind="dflash"`), `block_size 4`

## Why the KV cache is the binding constraint here, not throughput

16 full-attention layers × (K+V) × 4 KV heads × 256 head_dim × 2 bytes = **64 KiB per token**, paid once per copy. At `context_length 65536` that is 4.0 GiB for the live sequence, and every exact-APC snapshot pays it again, against 14.95 GiB of weights on a 40 GiB working set. On this machine a packed 4-bit KV is therefore not a decode optimization — it decides how much context is usable and how many conversations stay warm. That is the reason #2172 got read closely here rather than filed away.

`--kv-bits` is not that lever today: it was silently dense on any server with a drafter until 0.7.0 shipped #1822 (#2093, confirmed here from the safetensors headers of our own APC snapshots), and once it does engage, #2163 reports the uniform path peaking **higher** than f16 and OOMing at long context on this exact model — plus it never reaches the fused `head_dim 256` kernels.

## Condition 1: "no arbitrary array mask" excludes the whole server path, not just batches

`BatchKVCache.make_mask` (`mlx_vlm/models/cache.py:1187`) delegates to `create_causal_mask` (`mlx_vlm/models/cache.py:24`):

```python
def create_causal_mask(N, offset=0, window_size=None, right_padding=None, left_padding=None):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    ...
    mask = linds >= rinds
    ...
    return mask
```

It returns an `mx.array` **unconditionally** — there is no path on which it yields the `"causal"` string. With one sequence and no left padding the result is a plain causal bool array: semantically identical to `"causal"`, structurally an array. So a routing rule that excludes array masks excludes every request served through the batching generator, which is the server's only generation path. Not an edge case, the default case.

**This is the second time the same structural exclusion has hidden a kernel from this server.** mlx's fused `head_dim 256` full-attention path declined array masks; a local patch that forces it reported `fused` in our start banner while all 16 full-attention layers ran unfused, for weeks, because the batch cache passes nothing but array masks. Measured per layer once that was found, at `qL=2048 / kL=22747`: **2362 MiB unfused against 205 MiB fused, 83.9 ms against 68.7 ms**. The resolution upstream was not for callers to change their mask — it was ml-explore/mlx#4416 (merged 2026-09-02, still unreleased), which routes head-dim-256 prefill *with an array mask* to the fused kernel.

**Ask:** is array-mask acceptance a kernel limitation or a routing choice? If it is a routing choice, accepting the bool causal array that `create_causal_mask` already produces (or normalizing it centrally for the single-row, no-padding case) is what makes a packed-KV backend reachable from the server at all.

## Condition 2: "query length 1–4" versus what a drafter actually asks for

With speculative decoding the target never runs at `qL=1`; it runs verification at block + 1. At the DFlash 2 default `block_size 4` that is **qL=5**, and with the adaptive 3–5 row range documented for the upstream DFlash 2 path the target sees `qL` 4–6. A packed path bounded at 4 is therefore reachable only at drafter block ≤ 3.

That tension is resolvable — upstream's own DFlash 2 sweep (PR #2014, quoted in #2144) makes block 3 the *fastest* setting on this model: 55.64 tok/s at 3, 52.64 at 4, 49.56 at 5. But it should be documented rather than discovered per deployment, because the failure mode is a silent fallback, not an error.

**Ask:** is `qL ≤ 4` a hard kernel bound or a chosen one? If hard, drafter defaults and the packed-KV path need to reference each other in the docs.

## Condition 3, the cheapest one: make the decline observable

Both conditions above fail *silently* by design ("unsupported cases dequantize and use existing attention paths"). #2093 is what that costs: a server advertised 32 KiB/token for weeks while writing dense caches, and the only way to see it from outside was to range-read the safetensors headers of the snapshot files. A single log line or counter per process — `native KV path declined: reason=array_mask qL=5 layer=3`, once per distinct shape — turns a 3× regression from invisible into a grep.

## What we will measure once something exists to measure

Offered concretely, on the setup above, so the request is testable rather than a wish: for this geometry (24/4, `head_dim 256`, 16 full-attention layers of 64), with the batch path's own causal array mask, one sequence, and exact-APC snapshots restored from disk —

| shape | question |
|---|---|
| `qL` 1, 4, 5, 6 | does the packed path engage, per layer, per shape? |
| — | KV bytes/token actually stored (read back from the snapshot headers, not from a banner) |
| 8k / 32k / 55k | decode tok/s and peak working set, packed against f16 |

Happy to run that and post the numbers, including the negative results.
