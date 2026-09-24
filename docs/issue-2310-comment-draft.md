# Draft comment for upstream issue #2310

Status: **not posted, and now moot** — the issue was closed on 2026-09-23 when
PR #2328 merged. Kept for the measurements, which `upstream-2026-09-21.md` links.
Measured 2026-09-21 on `0.7.2`.

What it adds to the issue: the reporter measured batch 8/16/32 on a 128 GB M5
Max and saw 64 GB against an 11 GB working set. This is the other end of the
range — a single-slot and a four-slot server on a 48 GB machine, where the
cycle is present and provably removed by the fix, but **costs nothing
measurable** because the cyclic GC keeps up. Bounding a bug report downward is
worth as much as confirming it, and it tells the maintainer that a fix is
cheap-but-not-urgent rather than a release blocker.

---

## Confirmed on 0.7.2, and the fix works — but the impact is batch-size-bound

Confirming the mechanism on `0.7.2` (`a74c7de`) and adding the low-concurrency
end of the range, since the report covers batch 8 to 32.

**Setup:** Apple M5 Pro, 48 GB, macOS 26, `iogpu.wired_limit_mb=40960`,
`mlx-vlm 0.7.2`, `mlx 0.32.2`, `mlx-community/Qwen3.8-27B-4bit` (`qwen3_5`,
hybrid GDN + full attention, 64 layers, 16 of them full-attention), server with
continuous batching, APC disabled for these runs so that a snapshot cannot be
mistaken for a retention.

### The cycle is real on 0.7.2

Your model-free reproduction, adapted (12 layers, 4 rows, 2048 tokens,
head_dim 128, bf16, `gc.disable()`), against the installed package:

```
Runde 1: aktiv  0.47 GiB nach del
Runde 2: aktiv  0.94 GiB nach del
Runde 3: aktiv  1.41 GiB nach del
gc.collect(): 187 Objekte -> aktiv  0.00 GiB
```

Linear accumulation, fully reclaimed by one collection — exactly as described.
With the patch below the same script prints `0.00 GiB` after every round and
`gc.collect()` has 100 objects instead of 187 to look at.

### On a real server the collector keeps up

Idle floor of `mx.get_active_memory()` between requests, sampled once per
second from the server's own memory probe, every prompt unique so nothing is
served from a cache. Floor rather than peak: peaks track the prompt, only a
retention makes the floor climb.

**One slot** (`--max-num-seqs 1`, drafter on, 6 requests x ~8.1k tokens):

| | idle before | after 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| stock `0.7.2` | 15.96 | 17.12 | 17.12 | 17.12 | 17.12 | 17.12 | 17.12 |
| with the patch | 15.96 | 17.12 | 17.12 | 17.12 | 17.12 | 17.12 | 17.12 |

**Four slots** (`--max-num-seqs 4`, no drafter to keep #2033 out of the
measurement, 24 requests in 6 waves of 4 concurrent, ~9.0k tokens each), stock
`0.7.2`:

| idle before | wave 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 14.95 | 15.05 | 15.05 | 15.05 | 15.05 | 15.05 | 15.05 |

Peaks reached 24.7 GiB inside a wave and came all the way back. The 1.16 GiB
step at one slot is not the cycle: it survives the patch unchanged.

So at this batch size the cycle is collected before it can accumulate — a
request allocates enough to trigger generation-0 collections on its own. Your
numbers get worse with smaller `prefill_batch_size` precisely because `extend()`
fires more often, and at one slot `extend()` never fires at all: it needs a
non-empty batch to merge into. Only `filter()` runs here, once per finished
request.

### Patch

Breaking the self-reference is enough; `targets` is still captured, but nothing
points back at the function, so the closure dies with the call. Order changes
(a stack pops last-first) and does not matter — the list is only splatted into
`mx.eval()`.

```diff
--- a/mlx_vlm/generate/ar.py
+++ b/mlx_vlm/generate/ar.py
@@
-        def append_arrays(value):
-            if isinstance(value, mx.array):
-                targets.append(value)
-            elif isinstance(value, (list, tuple)):
-                for item in value:
-                    append_arrays(item)
+        def append_arrays(value):
+            stack = [value]
+            while stack:
+                item = stack.pop()
+                if isinstance(item, mx.array):
+                    targets.append(item)
+                elif isinstance(item, (list, tuple)):
+                    stack.extend(item)
```

Happy to open this as a PR if you want it; it is three lines and needs no test
beyond the model-free reproduction already in this issue.
