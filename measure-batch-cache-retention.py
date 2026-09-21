#!/usr/bin/env python3
"""Measure how much GPU memory stays allocated after a request has finished.

Upstream #2310: GenerationBatch._eval_pending_state builds a nested function
append_arrays that calls itself. The self-reference puts a cell for
append_arrays AND a cell for `targets` into the closure, so
function -> cell -> function is a reference cycle with every KV array of the
batch hanging off it. Refcounting cannot free that; the memory waits for
CPython's cyclic GC.

_eval_pending_state is reached from two places:

    extend()  when a batch is merged into a NON-EMPTY batch
    filter()  when len(keep) < len(uids), i.e. a sequence finished

At MAX_NUM_SEQS=1 the first never happens and the second happens once per
finished request, so this profile leaks one request's KV cache per request
until a collection runs. This script measures exactly that: the IDLE FLOOR of
`mem active` between requests, which is what the retained arrays raise.

Read the floor, not the peak. Peaks move with the prompt; the floor is what a
leak makes climb from request to request.

Run it with APC OFF. An exact-APC snapshot is a legitimate retention and would
show up in the same number:

    ENABLE_APC=0 MEM_PROBE_INTERVAL=1 ./start-mlx_qwen3.8.sh

    ./measure-batch-cache-retention.py                    # 6 x 8k tokens
    ./measure-batch-cache-retention.py --requests 10 --prompt-tokens 16384

Every prompt is unique, so nothing is served from a cache even if one is on.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

GiB = 1 << 30

MEM_RE = re.compile(
    r"mem active=(?P<active>[\d.]+) cache=(?P<cache>[\d.]+) sum=(?P<sum>[\d.]+) GiB"
    r".*?peak=(?P<peak>[\d.]+) GiB"
)


class LogWindow:
    """Reads the mem samples the server writes to its log, for one window."""

    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.mark()

    def mark(self):
        try:
            self.pos = os.path.getsize(self.path)
        except OSError:
            self.pos = 0

    def samples(self):
        """All (active, sum, peak) samples since the last mark(), in GiB."""
        out = []
        try:
            with open(self.path, "r", errors="replace") as fh:
                fh.seek(self.pos)
                for line in fh:
                    m = MEM_RE.search(line)
                    if m:
                        out.append(
                            (
                                float(m.group("active")),
                                float(m.group("sum")),
                                float(m.group("peak")),
                            )
                        )
        except OSError:
            pass
        return out


def make_prompt(tokens, salt):
    """~`tokens` tokens of unique text. ~0.75 words/token is close enough."""
    words = int(tokens * 0.75)
    head = f"Document {salt}, revision {salt * 7919}. "
    body = (
        f"Paragraph {{i}} of set {salt}: the memory planner reserves capacity "
        "before the prefill runs and releases it when the sequence finishes. "
    )
    parts, n = [head], 0
    i = 0
    while n < words:
        chunk = body.format(i=i)
        parts.append(chunk)
        n += len(chunk.split())
        i += 1
    return "".join(parts)


def ask(url, model, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return time.time() - t0, data.get("usage", {})


def _send_wave(args, wave):
    """One wave of `--concurrency` requests, all in flight at the same time."""
    prompts = [
        make_prompt(args.prompt_tokens, wave * 1000 + k) for k in range(args.concurrency)
    ]
    t0 = time.time()
    if args.concurrency == 1:
        wall, usage = ask(args.url, args.model, prompts[0], args.max_tokens, args.timeout)
        return wall, usage
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(ask, args.url, args.model, p, args.max_tokens, args.timeout)
            for p in prompts
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    usage = results[0][1] if results else {}
    return time.time() - t0, usage


def main():
    ap = argparse.ArgumentParser(
        description="Idle GPU memory floor between requests (upstream #2310)"
    )
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="Qwen3.8-27B-local")
    ap.add_argument("--log", default=os.path.expanduser("~/.mlx-qwen38/logs/server.log"))
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="requests in flight at once. >1 needs MAX_NUM_SEQS>1 on the server "
        "and is the only way to reach the extend() path, which merges into a "
        "non-empty batch. Mind #2033: concurrency plus a drafter corrupts "
        "without KV quantisation -- use ENABLE_SPEC_DECODE=0 for those runs.",
    )
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument(
        "--settle",
        type=float,
        default=8.0,
        help="idle seconds after each request; the floor is read from this window",
    )
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument(
        "--label", default="", help="free text for the header, e.g. 'with patch 0042'"
    )
    args = ap.parse_args()

    window = LogWindow(args.log)
    if not os.path.exists(args.log):
        print(f"  ⚠️  no log at {args.log} -- the memory columns stay empty", file=sys.stderr)

    print(f"\n  #2310 retention{' -- ' + args.label if args.label else ''}")
    print(f"  {args.requests} requests x ~{args.prompt_tokens} tokens, "
          f"max_tokens {args.max_tokens}, settle {args.settle:g}s\n")

    # Idle floor before anything ran: weights plus whatever startup left behind.
    window.mark()
    time.sleep(args.settle)
    pre = window.samples()
    base = min(s[0] for s in pre) if pre else None
    if base is None:
        print("  ⚠️  no mem samples -- is MEM_PROBE_INTERVAL=0?", file=sys.stderr)
    else:
        print(f"  idle before:  active {base:5.2f} GiB\n")

    label = "req" if args.concurrency == 1 else "wave"
    print(f"  {label:>4} {'prompt':>8} {'wall':>8} {'peak':>8} {'floor':>8} {'d(floor)':>9}")
    prev = base
    floors = []
    waves = max(1, args.requests // args.concurrency)
    for i in range(1, waves + 1):
        window.mark()
        try:
            wall, usage = _send_wave(args, i)
        except urllib.error.URLError as e:
            print(f"  ✗ request {i} failed: {e}", file=sys.stderr)
            return 1
        during = window.samples()
        peak = max((s[0] for s in during), default=None)

        window.mark()
        time.sleep(args.settle)
        idle = window.samples()
        floor = min((s[0] for s in idle), default=None)
        floors.append(floor)

        delta = (floor - prev) if (floor is not None and prev is not None) else None
        prev = floor if floor is not None else prev
        print(
            f"  {i:>4} {usage.get('prompt_tokens', '?'):>8} {wall:>7.2f}s "
            f"{_fmt(peak):>8} {_fmt(floor):>8} {_fmt(delta, sign=True):>9}"
        )

    good = [f for f in floors if f is not None]
    if len(good) >= 2 and base is not None:
        total = good[-1] - base
        per = total / len(good)
        print(
            f"\n  floor {base:.2f} -> {good[-1]:.2f} GiB over {len(good)} "
            f"{'requests' if args.concurrency == 1 else 'waves'}"
            f"  ({total:+.2f} GiB, {per:+.2f} GiB each)"
        )
        print(
            "  A floor that climbs with every request is the #2310 cycle: the KV\n"
            "  arrays of the finished request stay reachable until the cyclic GC\n"
            "  runs. A flat floor means the collector kept up (or the patch did)."
        )
    return 0


def _fmt(v, sign=False):
    if v is None:
        return "-"
    return f"{v:+.2f}" if sign else f"{v:.2f}"


if __name__ == "__main__":
    sys.exit(main())
