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
**`mlx-vlm 0.7.0rc0`**, `transformers 5.15.1`, `numpy 2.5.2`,
`huggingface-hub 1.27.0`, `pillow 12.3.0`, Python 3.12.

> Back on a tagged release since 2026-09-01 (previously main @`3fd38f4`).
> Install it with **`--no-deps`**:
>
> ```sh
> uv pip install --python ~/src/mlx/.venv/bin/python --no-deps "mlx-vlm==0.7.0rc0"
> ```
>
> Without `--no-deps` the resolver pulls `mlx` from PyPI down to 0.32.1, which
> silently disables patch `0013` -- see [docs/build-mlx.md](docs/build-mlx.md).
> All nine patches apply to this tag unchanged. Note the tag (`579cd51`) is
> behind main: `#1822` and the TurboQuant batch-decode fix are **not** in it.

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
| `APC_ENTRIES` | 1 | 2 | 3 |
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

A **cold** 30k prefill takes 2–3 minutes. That is why the prefix cache and the
SSD tier are a precondition rather than an optimisation: measured 89,630 ms →
350 ms for a 36k prompt after a restart.

---

## Patches

Nine patches against `site-packages`, applied by `patches/apply-patches.sh`
(idempotent, `--check` / `--revert`). **They vanish on every
`pip install -U mlx-vlm`** -- run it again afterwards.

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
- `0033` (upstream PR `#2072`) is new: the exact-APC snapshot store clones the
  live prompt cache, and that clone -- not the prefill -- is where three OOMs in
  two days actually happened. `APC_ENTRIES` on `roomy` went 3 -> 2 alongside it.

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

# APC disk tier, per namespace
du -sh ~/.mlx-qwen38/apc/*/

# Did the machine fall asleep mid-request?
pmset -g log | grep -E "Entering Sleep state|Wake Requests" | tail -5
```

| symptom | cause |
|---|---|
| `cached_tokens=0` in turn 2 | mlx-vlm < 0.6.13, or the snapshot was evicted (`APC_ENTRIES`) |
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
