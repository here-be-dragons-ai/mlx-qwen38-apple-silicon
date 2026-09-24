# Qwen3.8-27B (MLX 4bit) on Apple Silicon

Setup and start scripts for running **Qwen3.8-27B** as a local, OpenAI-compatible
server (`mlx-vlm`) on an Apple Silicon Mac.

The weights (14.95 GiB) are not the problem. The bottleneck is the KV cache at
**64 KiB per token**, paid once per copy (the running sequence plus every
prefix-cache snapshot). It decides how much context is left, and therefore
whether this is usable as an agent backend. The start script computes that
budget at runtime from the machine's real values and prints it.

| profile | target machine | peak RAM | `context_length` |
|---|---|---|---|
| `lean` | 32 GB **without** `sudo` | ~18.8 GiB | 32768 |
| `balanced` | 32 GB with `wired_limit 26624` | ~25.0 GiB | 49152 |
| `roomy` | 48 GB with `wired_limit 40960` | ~33 GiB | 65536 |

`PROFILE=auto` (default) picks one from the Metal working set. Tested on
macOS 26.

---

## Requirements

| | |
|---|---|
| Hardware | Apple Silicon (arm64), ≥ 32 GB unified memory |
| macOS | current, with Xcode Command Line Tools (`xcode-select --install`) |
| Disk | ~20 GB for model + drafter, plus up to 80 GB for the SSD prefix cache |
| Network | one-off ~15 GB download from HuggingFace |

Xcode.app plus the Metal toolchain is needed **only** for the optional mlx source
build ([docs/build-mlx.md](docs/build-mlx.md)). Everything runs without it.

---

## Installation

```sh
git clone https://github.com/here-be-dragons-ai/mlx-qwen38-apple-silicon.git
cd mlx-qwen38-apple-silicon

# 1. Software + model weights (idempotent, downloads resume)
./install-prereqs.sh

# 2. Raise the GPU wired limit -- the most important step
sudo ./set-iogpu-wired-limit.sh

# 3. Start the server (127.0.0.1:8888)
./start-mlx_qwen3.8.sh
```

On 32 GB, step 2 is the difference between ~23k and ~48k usable context. The
setup runs without it -- `PROFILE=auto` detects that and switches to `lean`.

`install-prereqs.sh` creates or verifies: Xcode CLT → [uv](https://astral.sh/uv)
→ venv under `~/src/mlx/.venv` (Python 3.12) → mlx-vlm + dependencies → Metal
self-test → patches → model + drafter → `~/.mlx-qwen38/{logs,apc}`.

| option | effect |
|---|---|
| `--check` | verify only, change nothing |
| `--skip-model` | software yes, 15 GB download no |
| `--latest` | newest instead of the pinned versions |

Paths via env: `MLX_HOME` (default `~/src/mlx`), `MLX_MODELS`, `PYTHON_VERSION`.

**Pinned, verified state:** `mlx 0.32.2`, `mlx-lm 0.31.3`,
**`mlx-vlm 0.7.2`**, `transformers 5.15.1`, `numpy 2.5.2`,
`huggingface-hub 1.27.0`, `pillow 12.3.0`, Python 3.12.

> **`0.7.1` is the one release this setup cannot run.** Skip it. It carries the
> APC redesign `#2182` but not its fix `#2262`, and the difference is upstream
> issue **`#2259`**: the prefill reserve was sized from the largest
> snapshot-bytes-per-token ratio the process had ever seen, as a monotonic
> maximum. This model's 48 GDN layers hold a fixed recurrent state that does not
> scale with tokens, so **one short prompt** sets that ratio ~10x too high, every
> later prefill over-reserves, and exact APC stops storing *and* restoring for
> the rest of the process lifetime -- with no error, only `memory_skips` rising
> in `/metrics`. This server's traffic is short agent turns, so on that tag the
> prefix cache dies within minutes.
>
> From 2026-09-17 to 09-21 the pin was therefore a git commit, main @ `548b09b`,
> installed with `--no-deps`. **`0.7.2` (2026-09-21) ends that** and the pin is a
> plain version again:
>
> ```sh
> uv pip install --python ~/src/mlx/.venv/bin/python "mlx-vlm==0.7.2"
> ```
>
> The upgrade is administrative. Diffed against the commit that had been
> running, `0.7.2` has **no differences at all** in `apc.py`,
> `apc_adapters.py`, `apc_coordinator.py`, `models/base.py`, `speculative/` and
> `server/generation.py` -- the whole APC and speculative surface, and
> everything the seven patches touch. All seven apply without fuzz.
>
> `--no-deps` is no longer needed: `0.7.2` declares `mlx>=0.32.2`, a lower
> bound, so the exact `mlx` pin survives the same resolution. Under the git
> requirement it did not, and patch `0013` fell inert when `mlx` slid to 0.32.1
> -- see [docs/build-mlx.md](docs/build-mlx.md).
>
> The start script still tests for the `#2259` **mechanism** (`_bytes_per_token`
> in `apc.py`) rather than the version, because the failure is silent and a
> future release can reintroduce a proportional estimate.
>
> Two things the tag does not fix: **`#2310`**, a KV-cache leak in
> `GenerationBatch._eval_pending_state` that fires once per finished request on
> this profile (measured harmless here; fixed on main by `#2328` on 09-23, not
> yet released), and **`#2239`**, the intermittent hang on requests carrying a
> `tools` array.

0.6.16 removed two long-standing constraints that still hold: DFlash 2 ships
upstream (PR #2014), and the ArraysCache buffer leak that killed generations at
~10.3k tokens is fixed (#1972 via PR #1984) -- verified here with 11,436 tokens
in one response, peak 18.56 GiB. `max_tokens` no longer needs the old 8192 cap.

`mlx 0.32.2` has been on PyPI since 2026-08-25 including `mlx-metal` and
`macosx_26_0_arm64` wheels, so **no source build is required** for the fused
`head_dim 256` path -- see [docs/build-mlx.md](docs/build-mlx.md) for the
history.

### Making the wired limit persistent

`sysctl -w` does not survive a reboot. Permanently:

```sh
sudo ./install-wired-limit-daemon.sh
```

The value is computed at boot from `hw.memsize` -- RAM minus 6 GiB (≤ 32 GB) or
8 GiB (above):

| RAM | `wired_limit_mb` | reserve |
|---|---|---|
| 32 GB | 26624 | 6 GiB |
| 48 GB | **40960** | 8 GiB |
| 64 GB | 57344 | 8 GiB |

The script clamps upwards, including for hand-passed values. **Do not set 45056
on 48 GB**: that leaves macOS ~2 GiB and produced a kernel panic on 2026-08-21,
without any warning beforehand -- macOS' `memoryPressure` evaluates the
compressor, not wired memory, and there is no jetsam for GPU wired memory.

Verify with `sysctl iogpu.wired_limit_mb`; preview with
`sudo /usr/local/libexec/set-iogpu-wired-limit.sh --dry-run`. Remove with
`--uninstall`.

> Without the limit set, `roomy` falls back to `PREFILL_STEP=512`, and the NAX
> path (which needs `qL >= 1024`) stops engaging. The start script says so
> explicitly.

---

## Operation

```sh
./start-mlx_qwen3.8.sh                        # PROFILE=auto
PROFILE=lean ./start-mlx_qwen3.8.sh           # minimal RAM, no sudo needed
PORT=8899 ./start-mlx_qwen3.8.sh              # lab instance
ENABLE_SPEC_DECODE=0 ./start-mlx_qwen3.8.sh   # without drafter
./watchdog-mlx_qwen3.8.sh                     # restarts before memory fills up
```

A profile sets *defaults* only; individual env variables still win.

| | `lean` | `balanced` | `roomy` |
|---|---|---|---|
| `APC_ENTRIES` | 1 | 2 | 2 |
| `KV_BITS` | 8 (from 8k tokens) | — (f16) | — (f16) |
| `PREFILL_STEP` | 512 | 1024 | 2048 |
| `VISION_CACHE` | 1 | 4 | 20 |
| `APC_DISK_MAX_GB` | 40 | 40 | 80 |
| `context_length` | 32768 | 49152 | 65536 |

| variable | default | effect |
|---|---|---|
| `PROFILE` | `auto` | `lean` / `balanced` / `roomy` |
| `KV_BITS` | per profile | `8` halves 64 → 32 KiB/token, doubles the context |
| `APC_ENTRIES` | per profile | prefix-cache snapshots = conversations kept warm |
| `APC_NS_KEEP_DAYS` | `3` | GC for inactive APC namespaces; `0` cleans now |
| `ENABLE_SPEC_DECODE` | `1` | drafter, +58…132% decode |
| `DRAFT_KIND` | `dflash` | `mtp` switches back ([docs/drafter.md](docs/drafter.md)) |
| `BIND_HOST` / `PORT` | `127.0.0.1` / `8888` | bind address |
| `MODEL_ALIAS` | `Qwen3.8-27B-local` | **must** match the model name in the request |
| `STATE_DIR` | `~/.mlx-qwen38` | log and SSD prefix cache |

> **`ENABLE_SPEC_DECODE=0` is an operating mode again since patch `0035`.**
> Without a drafter, a warm exact-APC hit used to decode at two thirds of the
> cold rate (upstream issue `#2210`): Qwen3.5's single-row shortcut ran
> `extract()` + `merge()` on every full-attention cache, i.e. copied the whole
> KV prefix on every decode token. `0035` is upstream PR `#2336`, which borrows
> the arrays instead. Measured 2026-09-24 at 26,690 tokens, 300 decoded tokens,
> no drafter: warm/cold **0.658 -> 0.998** (10.4-10.6 -> 16.0-16.1 tok/s), and
> the warm-hit memory spike (26.8-30.8 GiB) is gone (20.6 GiB). Greedy output is
> bit-identical with and without it. With the drafter nothing changes: every
> verify pass carries `capture_layer_ids`, which skips that shortcut entirely.
> Instrument: `./measure-apc-warm-decode.py`.

On start the script prints the computed budget of this machine. The
`CONTEXT BUDGET` line is an **upper bound, not a promise** -- the `mem` lines in
the log are authoritative. Details in [docs/memory.md](docs/memory.md).

> With mlx-vlm the `model` string from the request **is** the load path -- there
> is no `--alias`. On a mismatch the server discards the loaded model and starts
> a HuggingFace download (→ 401, even though the model is local). The script
> creates a symlink and warns on mismatch.

---

## Client configuration

OpenAI chat completions on `http://localhost:8888/v1`. Two rules: **model name =
alias** and **context ≤ budget**.

| setting | value | why |
|---|---|---|
| model name | `Qwen3.8-27B-local` | must match the symlink name |
| `base_url` | `http://localhost:8888/v1` | |
| `context_length` | `lean` 32768 · `balanced` 49152 · `roomy` 65536 | ≤ the budget from the start banner |
| `max_tokens` | 16384 | matches the client's response reserve, see below |
| `reasoning_effort` | `low` | only `low\|medium\|xhigh`, or `none`/`off` to disable thinking; anything else → HTTP 500 |

```yaml
model:
  default: Qwen3.8-27B-local
  base_url: http://localhost:8888/v1
  api_key: sk-local
  context_length: 65536
  max_tokens: 16384
  extra_body:
    enable_thinking: true
    reasoning_effort: low
compression:
  threshold: 0.85
```

**Why `max_tokens` is not simply maximised:** many clients trigger compaction at
`(context_length − max_tokens) × threshold`, so a larger `max_tokens` moves the
trigger down and wastes context. Pick the value your client reserves for a
response and no more.

The old hard cap of 8192 came from an upstream bug (#1972) that killed
generations at ~10.3k tokens; it is fixed since mlx-vlm 0.6.16.

The start script can cross-check a YAML config for you:

```sh
CLIENT_CONFIG=~/path/to/config.yaml ./start-mlx_qwen3.8.sh
```

It then warns when `model.context_length` exceeds the budget or `model.default`
does not match the alias. Without the variable the check is inert.

> **All clients must agree on `context_length`.** The overbooking guard sizes
> `APC_ENTRIES` from it; raising one client without lowering `APC_ENTRIES`
> silently caps the snapshot count to 1 and every client loses its warm slot.

---

## Speed to expect

Dense: every decode step reads ~15 GiB, so this is memory bandwidth.

| | M5 Pro / 48 GB (measured) | M5 base / 32 GB (estimated) |
|---|---|---|
| decode raw | 17.5–18.4 t/s | ~8–10 t/s |
| decode with spec-dec | 26.9–41.5 t/s | ~13–20 t/s |
| prefill | 420–470 t/s | ~180–250 t/s |

Those figures are short-context. At 28,590 tokens the same machine measured
15.5–16.0 tok/s without the drafter and 20.1–21.0 with it (`temperature 0`, 300
decoded tokens, 2026-09-10) — the drafter is worth about a third there, not the
+58…132% it is worth on short prompts, because acceptance drops to ~47%.

A **cold** 30k prefill takes 2–3 minutes. That is why the prefix cache and the
SSD tier are a precondition rather than an optimisation: measured 89,630 ms →
350 ms for a 36k prompt after a restart.

---

## Patches

Eight patches against `site-packages`, applied by `patches/apply-patches.sh`
(idempotent, `--check` / `--revert`). **They vanish on every
`pip install -U mlx-vlm`** -- run it again afterwards.

Seven are **local work**. One is a cherry-picked foreign PR again: `0035` is
upstream `#2336` and fixes `#2210` (see below). It comes out when upstream merges
it -- `apply-patches.sh` reports `CONFLICT` then.

The set shrank from eleven on 2026-08-28 when the APC redesign landed: `0021`
became obsolete (the separate generation loop it worked around is gone) and
`0030` was replaced by a start-script guard. `0040` went earlier, when DFlash 2
landed upstream.

What changed over 2026-09-02 to 09-04:

- `0013` (fused `head_dim 256`) had **never once fired on the server**. It
  declined array masks, and the batching generator's `BatchKVCache` passes down
  nothing else -- so the start banner read `fused` while all 16 full-attention
  layers ran unfused. Measured per layer at `qL=2048 / kL=22747`: 2362 MiB
  unfused against 205 MiB fused, 83.9 ms against 68.7 ms. The `FUSED_OK` probe
  was rewritten with it; it used to ask only whether mlx knows the `force_fused`
  argument, which says nothing about whether the path is taken.
- `0032` (upstream PR `#2096`, drafter priming under chunked prefill) came in on
  09-02 and went out again on 09-04. It does what it says -- acceptance 48.2% ->
  54.7% over 24 measured runs -- but decode throughput did not move and it held
  1.5-3 GiB of per-chunk hidden captures on long prompts. On a machine at 95% of
  its working set that is the wrong trade. `./measure-drafter-acceptance.py` is
  the instrument, and it stays.
- `0033` (upstream PR `#2072`) was added on 09-04 against three OOMs in two
  days: the exact-APC snapshot store cloned the live prompt cache, and that
  clone -- not the prefill -- was the call site. `APC_ENTRIES` on `roomy` went
  3 -> 2 alongside it.
- `0031` and `0034` are **gone** since `0.7.0` (2026-09-07): the release carries
  `#2152` and `#2090`, which do what they did.
- The same release carries `#1822`, and that one matters. Until it, `--kv-bits`
  was **silently ignored** on any server running a drafter with a single
  sequence -- which is this setup. Confirmed here from the safetensors headers
  of our own APC snapshots: all 16 full-attention layers stored dense while the
  banner advertised 32 KiB/token (upstream issue `#2093`). KV quantisation only
  became measurable with `0.7.0`, and it is still off by default -- see
  `docs/memory.md`.

What changed on 2026-09-17, moving to main @ `548b09b`:

- `0033` is **gone**. `#2072`, the PR it carried, was closed unmerged upstream
  on 2026-09-10, and `#2182` plus `#2262`
  bound the same peak from the other end: the manager sizes a snapshot before
  deciding, keeps at most `memory_max_bytes` resident and spills anything larger
  straight to disk, explicitly without a second clone. The rejects made the call
  -- 2 of 13 hunks fail against the `0.7.1` tag, **5 of 13 against main**, across
  all four files, and the `ar.py` hunk fails because upstream now contains it
  verbatim. Reanchoring five hunks onto a planner being rewritten weekly, for a
  peak the rewrite already bounds, is the wrong trade. What is genuinely lost:
  `#2072`'s per-layer requantise-on-restore and the no-copy promotion of a
  consumed single `KVCache` row.
- `0015` was **reanchored into another file**. Upstream lifted
  `_decode_quantized_linears_fused` out of `models/qwen3_5/language.py` into the
  shared `speculative/ops/linear.py` and dropped the `_qwen3_5_` prefix from the
  module attribute. The body is unchanged, so the patch is the same two lines in
  a new place -- but against the **old** anchor hunk 1 still applies while hunk 2
  does not, which would leave a tree that looks patched and has an inert switch.
  There is still no opt-out upstream, so the 9 GiB floor remains the default.
  Re-verified in production after the move: `mem active` 17.07 GiB after five
  requests with the drafter, against the 26.00 GiB the fusion costs.
- The other six applied unchanged, `0013` included -- re-probed through the
  patched entry point with the real server mask, fused at `qL` 512/1024/2048.
- New in the banner: **`APC resident cap`**. Since `#2182` a snapshot above
  `min(8 GiB, working_set/10)` is not kept in RAM but on SSD -- 4.0 GiB here,
  i.e. ~32,768 tokens per snapshot at two entries, *below* the `roomy`
  `context_length`. It is reported, not fed into the budget arithmetic; the long
  note above `budget()` in the start script records why that was tried and
  reverted.

What changed on 2026-09-24:

- **`0035` added** (upstream PR `#2336`, open). Fixes `#2210` for runs without
  a drafter; a no-op with one. See the `ENABLE_SPEC_DECODE=0` note above and
  [docs/upstream-2026-09-24.md](docs/upstream-2026-09-24.md).

What changed on 2026-09-21, moving to the `0.7.2` release:

- **Nothing in the patch set.** The same seven patches, applied to `a74c7de`
  without fuzz. The files they touch are byte-identical to main @ `548b09b`,
  so the 09-17 verification still stands and none of it was re-measured.
- The install got simpler: plain version pin, no separate `--no-deps` step.

```sh
./patches/apply-patches.sh --check
```

Each patch is documented in the header of that script: origin, the measurement
that justifies it, and its rollback switch. That is the authoritative place; it
is not duplicated here.

---

## Diagnostics

```sh
# Is the prefix cache hitting? (turn 2 must show cached_tokens > 0)
rg "Prefill completed" ~/.mlx-qwen38/logs/server.log | tail -5

# Memory over time (WARNING from 85%)
rg "mem active" ~/.mlx-qwen38/logs/server.log | tail -20

# Patch status
./patches/apply-patches.sh --check

# Did the cache get REUSED, not just written? (new since upstream #2270)
curl -s localhost:8888/metrics | python3 -m json.tool | rg "stored_tokens|restored_tokens|memory_skips"

# APC disk tier, per namespace
du -sh ~/.mlx-qwen38/apc/*/

# Did the machine fall asleep mid-request?
pmset -g log | grep -E "Entering Sleep state|Wake Requests" | tail -5
```

| symptom | cause |
|---|---|
| `cached_tokens=0` in turn 2 | mlx-vlm < 0.6.13, or the snapshot was evicted (`APC_ENTRIES`) |
| `cached_tokens=0` from the first short prompt onwards, `memory_skips` rising | a downgrade to the `0.7.1` **tag** (upstream #2259) -- needs `0.7.2`, see above |
| `cached_tokens=1` on large prompts | mlx-vlm < 0.6.14 (short-prompt bug, PR #1901) |
| HTTP 401 / HF download on a request | model name ≠ alias symlink |
| HTTP 500 on every request | `reasoning_effort` outside the accepted set |
| `[METAL] Insufficient Memory` | context over budget → lower `context_length` or set `KV_BITS=8` |
| `APC_ENTRIES N -> capped to M` | `context_length` too high for the working set |

**On battery the Mac falls asleep mid-generation.** The log then reports
plausible `elapsed`/`rate` values while the wall clock jumps by minutes. Put
`caffeinate -dimsu` in front of measurements and long agent runs.

A `[METAL] Insufficient Memory` stack trace shows the **location** of the next
allocation, never the cause. Diagnostic order: `ENABLE_APC=0` to isolate, then
clear the SSD tier, and only then touch `APC_ENTRIES` or `context_length`.

---

## Documentation

| | |
|---|---|
| [docs/memory.md](docs/memory.md) | sizing, the memory investigation, what was ruled out and why |
| [docs/drafter.md](docs/drafter.md) | DFlash 2 vs MTP, measurements, patch dependency |
| [docs/build-mlx.md](docs/build-mlx.md) | building mlx 0.32.2 for fused `head_dim 256` |
| [docs/flash-next.md](docs/flash-next.md) | Qwen3.8-Flash-Next (177B) on 48 GB: how to spot a broken conversion, external PLE, expert offloading, why it lands at 4 tok/s |
| `patches/apply-patches.sh` | the patch set, each with its measurement |

---

## Files

| file | purpose |
|---|---|
| `install-prereqs.sh` | complete setup from a fresh macOS, idempotent |
| `start-mlx_qwen3.8.sh` | server start, profiles, live budget calculation |
| `watchdog-mlx_qwen3.8.sh` | restarts the server before memory fills up |
| `download-mlx-model.sh` | resumable HuggingFace downloader, with size check |
| `convert-dflash2-drafter.py` | quantizes the DFlash 2 drafter (bf16 → 4bit) |
| `measure-drafter-acceptance.py` | acceptance rate across the chunked-prefill boundary (patch `0032`) |
| `measure-batch-cache-retention.py` | idle GPU-memory floor between requests -- does a finished request give its KV cache back? (upstream #2310) |
| `measure-apc-warm-decode.py` | decode rate on an exact-APC warm hit against a cold request (issue `#2210`) |
| `set-iogpu-wired-limit.sh` | computes `iogpu.wired_limit_mb` from `hw.memsize`, clamps |
| `install-wired-limit-daemon.sh` | installs helper + LaunchDaemon, idempotent |
| `patches/apply-patches.sh` | apply / check / revert patches |

---

## Where the numbers come from

All values marked "measured" come from an **M5 Pro / 48 GB** machine (mlx-vlm
0.6.13/0.6.15, Qwen3.8-27B-4bit, `temperature=0`). The memory calculation is
arithmetic from `config.json` and the file sizes and holds on any machine; the
throughput figures for 32 GB are estimates scaled via memory bandwidth, and
marked as such.

---

## License

[MIT No Attribution](LICENSE) (SPDX: `MIT-0`) -- copy, adapt and reuse without
attribution.
