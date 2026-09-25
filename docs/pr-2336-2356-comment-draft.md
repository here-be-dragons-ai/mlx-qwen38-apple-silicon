# Draft comments for upstream PRs #2336 and #2356

Status: **not posted.** Measured 2026-09-24 (0.7.2) and 2026-09-25 (0.7.3).
Both PRs ask for independent measurements; these are from a different machine
and a real 27B checkpoint. Post the first under
https://github.com/Blaizzy/mlx-vlm/pull/2336 and the second under
https://github.com/Blaizzy/mlx-vlm/pull/2356, or merge them into one comment on
`#2210` that links both.

---

## For #2336

Independent measurement on a real checkpoint, since the PR deliberately claims
no throughput number: on this setup it is the fix for #2210.

**Setup:** Apple M5 Pro 48 GB, macOS 26, mlx 0.32.2, mlx-vlm 0.7.3 (also
measured on 0.7.2), `mlx-community/Qwen3.8-27B-4bit` (`qwen3_5`, 16
full-attention layers), server with continuous batching, **one** sequence,
exact APC with disk tier, f16 KV, **no drafter**. 26,690-token prompt, 300
decoded tokens, `temperature 0`, pairs of identical prompts where only the cold
arm carries a nonce, one server restart per arm.

| arm | cold tok/s | warm (APC hit) tok/s | warm/cold | warm `active+cache` |
|---|---:|---:|---:|---:|
| 0.7.3 | 16.08 / 16.08 | 10.49 / 10.57 | **0.655** | 28.8 / 26.8 GiB |
| 0.7.3 + this PR | 16.02 / 16.04 | 16.05 / 16.06 | **1.001** | 22.2 / 22.2 GiB |

0.7.2 gave the same picture (0.658 → 0.998). Greedy output is bit-identical with
and without the PR (4k prompt, 250 tokens, cold and warm).

Two observations that may help review:

1. **The gate matches more than "a batch that shrank to one".**
   `_is_single_row_batch_cache` matches any single-row, unquantized
   `BatchKVCache`, so on a single-slot server every request restored from APC
   takes the extract/merge path on every decode token — which is exactly
   #2210. Note that `extract()` returns an exactly-sized `KVCache`, so the
   following `update_and_fetch` must reallocate and concatenate as well: three
   passes over the prefix per layer per step, not two. In isolation (16 layers
   × 4 KV heads × 256, bf16, one token) that is 10.5 ms at 8k, 29.5 ms at 26.7k
   and 57.9 ms at 50k per forward, against under 1 ms borrowed.
2. **Speculative decoding never reaches this branch.** The shortcut requires
   `hidden_sink is None`, and DFlash/MTP verify passes always pass
   `capture_layer_ids`, so `hidden_sink` is a list. With DFlash 2 loaded the
   warm/cold ratio was 1.00 before this PR and stays there (cold 21.6–22.5 tok/s
   with the PR, unchanged). That explains why #2210 disappears as soon as a
   drafter is on, which I had reported there without knowing why.

---

## For #2356

Independent check on other hardware, as requested: it reproduces your result.

**Setup:** Apple M5 Pro 48 GB (not an M2 Ultra), macOS 26, mlx 0.32.2, mlx-vlm
0.7.3 + this PR (library change only, tests not installed),
`mlx-community/Qwen3.8-27B-4bit` unmodified, server, one sequence, exact APC
with disk tier, f16 KV, no drafter. 26,690-token restored prefix — about twice
your longest row — 300 decoded tokens, `temperature 0`, one restart per arm.

| arm | cold tok/s | restored tok/s | restored/cold | warm `active+cache` |
|---|---:|---:|---:|---:|
| 0.7.3 | 16.08 / 16.08 | 10.49 / 10.57 | **0.655** | 28.8 / 26.8 GiB |
| 0.7.3 + #2356 | 15.84 / 16.06 | 15.82 / 16.12 | **1.001** | 20.6 / 20.6 GiB |
| 0.7.3 + #2336 (for comparison) | 16.02 / 16.04 | 16.05 / 16.06 | **1.001** | 22.2 / 22.2 GiB |

Greedy output is bit-identical across all three arms. The memory column is the
part your table doesn't show: the restored arm on 0.7.3 sits 3–8 GiB above the
cold one at this length, and this PR removes that too.

On your question whether the batch caches on this path are intentional: #2336
fixes the same symptom inside the model (it borrows the arrays of a one-row
`BatchKVCache` instead of extract/merge), and it additionally covers a batch
that *shrinks* to one row, which `merge_rows` never sees. The two look
complementary rather than competing. With a speculative drafter neither
matters: the verify pass carries `capture_layer_ids`, which skips the
single-row shortcut entirely.
