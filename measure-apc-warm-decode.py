#!/usr/bin/env python3
"""Measure DECODE throughput on an exact-APC warm hit against a cold request.

The point of this script is upstream issue #2210. It reports that
`make_warm_batch_exact_cache_multi` merges the stored snapshot into batch-aware
caches even when there is exactly ONE row, and that Qwen3_5 then copies the
whole KV + recurrent state per decode token -- O(context) work per token. On
0.6.17, at 33k tokens: 6.5 tok/s warm against 9.9 tok/s cold, footprint 36 GiB
against 28 GiB. The reporter did not re-measure on 0.7.0.

Checked here on 0.7.0 by reading the source: the merge-first structure is
present and has no single-row shortcut (apc.py:3949, called from
apc_coordinator.py:119 and generate/ar.py:2700). Whether the per-token copy
survived the release is what this script decides.

WHY THIS IS NOT VISIBLE IN THE NUMBERS WE ALREADY HAVE: every APC measurement
in this repository is a PREFILL measurement (89,630 ms -> 350 ms). A warm hit
that decodes at two thirds of the cold rate still looks like a spectacular win
in those numbers, because the prefill it saves is worth minutes and the decode
it costs is worth seconds per turn -- until the answer is long, which for an
agent backend it always is.

    ./measure-apc-warm-decode.py                        # 2 pairs at 30k
    ./measure-apc-warm-decode.py --prompt-tokens 33000 --repeat 3
    ./measure-apc-warm-decode.py --max-tokens 500       # longer decode window

WHAT THE ARMS ARE. Each pair sends the SAME prompt twice:

    cold  a fresh nonce prefix, so no snapshot can match  -> cache_n == 0
    warm  the identical prompt again                      -> cache_n ~ the prompt

READ THE CONTEXT AS prompt_n + cache_n, NOT prompt_n. On a hit, prompt_n is
only the remainder that still had to be prefilled -- measured here as 1 token
against cache_n 1985, i.e. the snapshot came back one token short of the prompt,
not the 16 that patch 0010's guard produced on 0.6.x. A cold arm that reports
cache_n > 0, a warm arm that reports cache_n == 0, or a pair whose two arms
decode at different context lengths is a broken measurement, and is called out
as one instead of being averaged into a rate.

Cold necessarily precedes warm inside a pair -- the snapshot has to exist. Drift
is therefore controlled BETWEEN pairs, not inside them: run at least two and
compare the pairs against each other before believing the ratio.

CAREFUL, THE SERVER IS THE MEASUREMENT. It imports mlx_vlm once, at start.
Applying or reverting a patch under a running server changes nothing about what
that process executes; restart between arms. Put `caffeinate -dimsu` in front
of the server, or the machine sleeps mid-generation and the rates are fiction.
"""

import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))

# The filler pool is part of the instrument -- see the comment on it in
# measure-drafter-acceptance.py. One source for it, imported through a path
# because the file name has hyphens in it.
_SIBLING = os.path.join(_HERE, "measure-drafter-acceptance.py")


def _load_prompt_builder():
    spec = importlib.util.spec_from_file_location("_mda", _SIBLING)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the prompt builder from {_SIBLING}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MEM = re.compile(
    r"mem active=(?P<active>[\d.]+) cache=(?P<cache>[\d.]+) sum=(?P<sum>[\d.]+) GiB"
    r".*?peak=(?P<peak>[\d.]+) GiB"
)


class LogWindow:
    """Reads the mem samples the server logs every 5 s, for one request window."""

    def __init__(self, path):
        self.path = path
        self.offset = None

    def mark(self):
        try:
            self.offset = os.path.getsize(self.path)
        except OSError:
            self.offset = None

    def collect(self):
        """(max sum, last peak) in GiB over the window, or (None, None)."""
        if self.offset is None:
            return None, None
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read().decode("utf-8", "replace")
        except OSError:
            return None, None
        sums, peaks = [], []
        for m in _MEM.finditer(chunk):
            sums.append(float(m.group("sum")))
            peaks.append(float(m.group("peak")))
        if not sums:
            return None, None
        return max(sums), peaks[-1]


def ask(url, model, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    data["_wall_s"] = time.monotonic() - t0
    return data


def run_arm(args, prompt, window):
    window.mark()
    data = ask(args.url, args.model, prompt, args.max_tokens, args.timeout)
    mem_sum, mem_peak = window.collect()
    t = data.get("timings") or {}
    usage = data.get("usage") or {}
    prompt_n = t.get("prompt_n", usage.get("prompt_tokens", 0))
    cache_n = t.get("cache_n", 0)
    return {
        "prompt_n": prompt_n,
        "cache_n": cache_n,
        # What the decode actually runs against: freshly prefilled + restored.
        "ctx": prompt_n + cache_n,
        "prefill_s": (t.get("prompt_ms") or 0) / 1000.0 or None,
        "decode_tps": t.get("predicted_per_second"),
        "out_n": t.get("predicted_n", usage.get("completion_tokens", 0)),
        "accept": (
            100.0 * t["draft_n_accepted"] / t["draft_n"]
            if t.get("draft_n")
            else None
        ),
        "wall_s": data["_wall_s"],
        "mem_sum": mem_sum,
        "mem_peak": mem_peak,
    }


def _fmt(v, spec="6.1f", dash="     -"):
    return dash if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="Qwen3.8-27B-local")
    ap.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/src/mlx/models/Qwen3.8-27B-MLX-4bit"),
        help='tokenizer source for exact prompt lengths; "" to skip',
    )
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=30000,
        help="context under test; #2210 measured at 33k",
    )
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=2, help="cold/warm pairs")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument(
        "--log",
        default=os.path.expanduser("~/.mlx-qwen38/logs/server.log"),
        help='server log for the mem samples; "" to skip the memory column'
    )
    args = ap.parse_args()

    mda = _load_prompt_builder()
    tokenizer = mda.load_tokenizer(args.model_dir)
    window = LogWindow(args.log) if args.log else LogWindow(os.devnull)

    print(f"  endpoint : {args.url}  model={args.model}")
    print(
        f"  under test: exact-APC warm hit vs cold, {args.prompt_tokens} prompt "
        f"tokens, {args.max_tokens} decoded, temperature 0"
    )
    print(f"  pairs    : {args.repeat}\n")
    print(
        f"  {'pair':>4} {'arm':<5} {'ctx':>7} {'prompt_n':>9} {'cache_n':>8} "
        f"{'prefill_s':>10} {'decode t/s':>11} {'accept%':>8} {'mem_sum':>8} "
        f"{'peak':>7}"
    )
    print("  " + "-" * 88)

    pairs, broken = [], 0
    for pair in range(1, args.repeat + 1):
        nonce = f"[run {time.time_ns()}] "
        prompt = nonce + mda.build_prompt(args.prompt_tokens, tokenizer)

        results = {}
        for arm in ("cold", "warm"):
            try:
                r = run_arm(args, prompt, window)
            except urllib.error.URLError as exc:
                print(f"  {pair:>4} {arm:<5}   request failed: {exc}")
                broken += 1
                results = {}
                break
            results[arm] = r
            print(
                f"  {pair:>4} {arm:<5} {r['ctx']:>7} {r['prompt_n']:>9} "
                f"{r['cache_n']:>8} "
                f"{_fmt(r['prefill_s'], '10.2f', '         -')} "
                f"{_fmt(r['decode_tps'], '11.2f', '          -')} "
                f"{_fmt(r['accept'], '7.1f', '      -')}% "
                f"{_fmt(r['mem_sum'], '8.2f', '       -')} "
                f"{_fmt(r['mem_peak'], '7.2f', '      -')}"
            )

        if len(results) != 2:
            continue

        cold, warm = results["cold"], results["warm"]
        if cold["cache_n"]:
            print(
                f"       ^ BROKEN PAIR: the cold arm hit the cache "
                f"(cache_n={cold['cache_n']}). The nonce is not doing its job."
            )
            broken += 1
            continue
        if not warm["cache_n"]:
            print(
                "       ^ BROKEN PAIR: the warm arm did NOT hit "
                "(cache_n=0). Nothing about the warm path was measured -- check "
                "ENABLE_APC, APC_ENTRIES and whether the snapshot was evicted."
            )
            broken += 1
            continue
        if abs(cold["ctx"] - warm["ctx"]) > 2:
            print(
                f"       ^ BROKEN PAIR: the arms decoded at different context "
                f"lengths ({cold['ctx']} vs {warm['ctx']}). A rate ratio across "
                f"different kL says nothing."
            )
            broken += 1
            continue
        pairs.append((cold, warm))

    if not pairs:
        print("\n  no usable pair. Nothing measured.")
        return 1

    print()
    ratios = []
    for i, (cold, warm) in enumerate(pairs, 1):
        c, w = cold["decode_tps"], warm["decode_tps"]
        if not c or not w:
            continue
        ratios.append(w / c)
        dmem = (
            f", mem {cold['mem_sum']:.2f} -> {warm['mem_sum']:.2f} GiB"
            if cold["mem_sum"] and warm["mem_sum"]
            else ""
        )
        print(
            f"  pair {i}: warm/cold decode = {w:.2f}/{c:.2f} = {w / c:.3f}"
            f"  ({(w / c - 1) * 100:+.1f}%){dmem}"
        )

    if ratios:
        med = statistics.median(ratios)
        print(f"\n  median warm/cold decode ratio: {med:.3f}")
        if med < 0.9:
            print(
                "  #2210 REPRODUCES on this version: the warm hit decodes "
                f"{(1 - med) * 100:.0f}% slower than the cold request at the same\n"
                "  context. That is the per-decode-token cache copy, and it is paid "
                "on every turn an agent takes."
            )
        elif med > 1.1:
            print(
                "  Inverted: the warm hit decodes FASTER. The ctx column already "
                "rules out the\n  obvious explanation, so this is a real "
                "difference in how the two caches were built."
            )
        else:
            print(
                "  #2210 does NOT reproduce here: warm and cold decode within 10% "
                "of each other.\n  The 0.7.0 release closed the per-token copy, or "
                "this configuration never takes that path."
            )
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
