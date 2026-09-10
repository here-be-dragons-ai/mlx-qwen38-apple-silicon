<!-- posted 2026-09-10 to https://github.com/Blaizzy/mlx-vlm/issues/2210 (@felix-ab)
     comment: https://github.com/Blaizzy/mlx-vlm/issues/2210#issuecomment-5623370125
     author: gtonic (the account authenticated in `gh`)
     Measured 2026-09-10. Raw output of the four arms is quoted verbatim below;
     the instrument is ./measure-apc-warm-decode.py in this repository. -->

Re-measured on **0.7.0**, which you flagged as the open question. It reproduces — but only with speculative decoding **off**, and that turned out to be the interesting part.

## Setup

- Apple M5 Pro, 48 GB, macOS 26, `iogpu.wired_limit_mb=40960` (40 GiB working set)
- mlx-vlm 0.7.0, mlx + mlx-metal 0.32.2, mlx-lm 0.31.3
- `mlx-community/Qwen3.8-27B-4bit` (`qwen3_5`, hybrid GDN + full attention, 64 layers, 16 full-attention) — the 4-bit affine conversion, not your 6-bit one
- Server, continuous batching, **one** sequence, exact APC with disk tier, f16 KV (no `--kv-bits`)
- 28,590-token prompt, 300 decoded tokens, `temperature 0`, `caffeinate -dimsu`

Method: each pair sends the same prompt twice. The cold arm carries a nonce prefix so nothing can match; the warm arm is the identical prompt. Context is read as `prompt_n + cache_n`, because on a hit `prompt_n` is only the unmatched remainder (1 token here). A pair whose arms differ in context, or whose cold arm hits, is discarded rather than averaged.

## Four arms

| # | speculative decoding | local patches in the APC path | cold tok/s | warm tok/s | warm/cold |
|---|---|---|---:|---:|---:|
| A | DFlash 2, `block_size 4` | applied | 20.44 / 20.11 / 20.80 | 20.69 / 20.07 / 20.87 | **1.003** |
| B | DFlash 2, `block_size 4` | reverted | 20.73 / 20.90 | 20.81 / 21.01 | **1.004** |
| C | off | applied | 15.96 / 15.73 | 10.28 / 10.16 | **0.645** |
| D | off | reverted | 15.73 / 15.50 | 9.58 / 10.03 | **0.628** |

Arms B and D revert the two local patches of ours that touch this code path at all (one suppresses the redundant second exact snapshot per request, one carries #2072's ownership transfer). They change nothing about the result, which is why I am comfortable reporting it as upstream behaviour: **−35% to −39% on a warm hit, without a drafter, on 0.7.0.**

With a drafter the effect is gone — 1.003 and 1.004 across five pairs, i.e. inside the run-to-run spread. If the per-decode-token copy were merely amortised over a verification block, at `block_size 4` and 47% acceptance (~2.9 accepted tokens per target pass) a 35% per-token penalty should still show up as roughly −15%. It does not show up at all, which suggests the speculative path does not take that shortcut rather than paying it less often. I have not read far enough into it to state that as fact.

## The structural half of your report is still true on 0.7.0

`make_warm_batch_exact_cache_multi` (`mlx_vlm/apc.py:3949`) still merges unconditionally — there is no `len(row_caches) == 1` shortcut — and it is reached from `apc_coordinator.py:119` and `generate/ar.py:2700`. What I could not locate on 0.7.0 is the second half: `extract_prompt_cache_from_batch` has exactly one caller in the release, and it sits inside `apc.py` rather than in a model decode path.

## Footprint

Directionally consistent with your 36 vs 28 GiB, though noisier than the throughput. Peak `sum` observed inside the request window, per arm: the cold arms sat at 23.1–25.2 GiB, and two of the four warm arms spiked to **33.8** and **34.9 GiB** against a 40 GiB working set, while the other two stayed at 21.5 and 23.4. On a machine that runs at 95% of its working set that spread is its own problem, independent of the tok/s.

## Why this may have gone unnoticed

Every APC number I had before today was a *prefill* number (89,630 ms → 350 ms on a warm hit at 36k). A warm hit that decodes at two thirds of the cold rate still looks like an unambiguous win in that metric, because the prefill it saves is worth minutes per turn and the decode it costs is worth seconds — until the answer is long. And with a drafter on, which is the default for this model here, the decode penalty is not there at all.

Happy to run 33k, other block sizes, or an `--kv-bits` arm if any of those would help pin the path down.
