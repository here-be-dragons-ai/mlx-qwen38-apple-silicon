# Speculative decoding: DFlash 2 vs MTP

Background and measurements for the drafter. The README states only that DFlash 2
is the default and how to switch back.

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

---

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

---

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

---

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
