#!/usr/bin/env python3
"""Measure the DFlash 2 acceptance rate across the chunked-prefill boundary.

The point of this script is patch 0032 (upstream PR #2096). Chunked prefill
requested the speculative capture kwargs only on the FINAL prefill call, so a
drafter that reads target hidden past the last prompt position was primed from
a one-token prompt. DFlash 2 does exactly that: it cross-attends its first
draft block over the prompt hidden.

The damage is therefore invisible in the decode rate alone and only shows up
against an UNCHUNKED control -- a prompt shorter than PREFILL_STEP goes through
in one call and was never affected. That is why this script sweeps prompt
lengths around PREFILL_STEP instead of measuring a single one.

Upstream measured on Qwen3.5-9B with prefill_step_size 512 (% of drafted
tokens accepted):

    prompt   unchunked   chunked, before   chunked, after
      2048       82.3%             57.1%            82.3%
      4096       64.9%             55.4%            82.3%

Usage:
    ./measure-drafter-acceptance.py                         # default sweep
    ./measure-drafter-acceptance.py --prompt-tokens 1024 4096 16384
    ./measure-drafter-acceptance.py --repeat 3 --max-tokens 400

A/B PROCEDURE -- and the trap in it: the server imports mlx_vlm ONCE, at start.
Applying or reverting a patch under a running server changes nothing about what
that process executes. Both halves need a restart:

    ./patches/apply-patches.sh --revert   # or: remove 0032 and re-apply
    ./start-mlx_qwen3.8.sh                # restart, then measure
    ./patches/apply-patches.sh            # restore, restart, measure again

The banner below prints what the LOCAL venv contains, which is a statement about
the file on disk, not about the running server. Restart, then trust it.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# CAREFUL, THE FILLER IS PART OF THE INSTRUMENT. The first version of this
# script repeated ONE sentence to length. Every prompt then measured 100 %
# acceptance at every length: a drafter predicts a text it has already seen
# fifty times perfectly, so the defect under test cannot show through. The pool
# below is cycled with a deterministic stride so neighbouring sentences differ
# and the local continuation stays genuinely uncertain.
_FILLER_POOL = [
    "The cable car ascends the north face in twelve minutes, carrying eight passengers per cabin.",
    "Maintenance replaced the grip spring on carrier nineteen after the winter inspection found play.",
    "Wind at the top station gusted to forty-one kilometres per hour, below the sixty-five limit.",
    "The drive station logs every departure with a timestamp, a load estimate, and a tower reading.",
    "Rope tension drifted by two percent over the season, which the tensioning weight absorbed.",
    "A power interruption on the third of March moved forty passengers to the diesel backup drive.",
    "The evacuation drill took nineteen minutes for a fully loaded cabin at mid-span.",
    "Sensor four reported an intermittent fault that turned out to be a corroded connector.",
    "Snow load on the station roof reached one hundred and eighty kilograms per square metre.",
    "Ticketing recorded eleven thousand ascents in February, a quarter of them on three days.",
    "The gearbox oil analysis showed iron particles within tolerance but trending upward.",
    "Night maintenance realigned the return sheave after a bearing was replaced on tower six.",
]
# Asks for a long, generative answer. The earlier version asked for one sentence
# and got twenty tokens -- far too few drafted tokens for a rate to mean anything.
_QUESTION = (
    "\n\nBased only on the operations log above, write a detailed maintenance and "
    "safety briefing for the incoming shift. Cover the mechanical findings, the "
    "weather exposure, the power event, and what you would inspect first and why. "
    "Write at least 250 words in continuous prose."
)


def _filler(n_sentences: int) -> str:
    """Pool cycled with a stride coprime to its length, so neighbours differ."""
    m = len(_FILLER_POOL)
    return " ".join(_FILLER_POOL[(i * 5) % m] for i in range(n_sentences))


def build_prompt(target_tokens: int, tokenizer=None) -> str:
    """A prompt of roughly *target_tokens* tokens, exact when a tokenizer is given."""
    if tokenizer is None:
        # 1.23 tokens per word, MEASURED on this pool with the Qwen3.8 tokenizer
        # -- not the 0.75 that gets quoted for average English prose. The whole
        # measurement hinges on which side of PREFILL_STEP a prompt lands, so
        # pass --model-dir and let the tokenizer do it exactly whenever you can.
        words_needed = max(1, int(target_tokens / 1.23))
        words = _filler(words_needed).split()[:words_needed]
        return " ".join(words) + _QUESTION

    tail = tokenizer.encode(_QUESTION, add_special_tokens=False)
    budget = max(1, target_tokens - len(tail))
    # Overshoot, then trim to the exact token budget.
    text = _filler(budget // 4 + len(_FILLER_POOL))
    ids = tokenizer.encode(text, add_special_tokens=False)[:budget]
    return tokenizer.decode(ids) + _QUESTION


def load_tokenizer(model_dir):
    if not model_dir:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_dir)
    except Exception as exc:  # noqa: BLE001 - informational only
        print(f"  (no tokenizer from {model_dir}: {exc}; using the word estimate)")
        return None


def ask(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="Qwen3.8-27B-local")
    ap.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/src/mlx/models/Qwen3.8-27B-MLX-4bit"),
        help="tokenizer source for exact prompt lengths; \"\" to skip",
    )
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192],
        help="sweep; keep at least one value BELOW PREFILL_STEP as the "
        "unchunked control",
    )
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    try:
        import mlx_vlm.speculative.utils as spec

        patched = hasattr(spec, "splice_prompt_hidden")
    except Exception:  # noqa: BLE001
        patched = None

    state = {True: "APPLIED", False: "not applied", None: "unknown"}[patched]
    print(f"  venv     : patch 0032 {state}  (the SERVER must have been restarted)")
    print(f"  endpoint : {args.url}  model={args.model}")
    print(f"  sampling : temperature 0, max_tokens {args.max_tokens}\n")

    tokenizer = load_tokenizer(args.model_dir)

    print(
        f"{'asked':>7} {'prompt_n':>9} {'cached':>7} {'drafted':>8} "
        f"{'accepted':>9} {'accept%':>8} {'rounds':>7} {'tok/s':>7}"
    )
    print("  " + "-" * 68)

    failures = 0
    for want in args.prompt_tokens:
        for run in range(args.repeat):
            # A nonce per request, otherwise the APC serves the prefix back and
            # the prefill under test never happens (watch the cached column).
            nonce = f"[run {time.time_ns()}] "
            prompt = nonce + build_prompt(want, tokenizer)
            try:
                data = ask(args.url, args.model, prompt, args.max_tokens, args.timeout)
            except urllib.error.URLError as exc:
                print(f"{want:>7}   request failed: {exc}")
                failures += 1
                continue

            t = data.get("timings") or {}
            drafted = t.get("draft_n")
            accepted = t.get("draft_n_accepted")
            if drafted is None:
                print(
                    f"{want:>7} {t.get('prompt_n', '?'):>9} "
                    f"{t.get('cache_n', '?'):>7}   no drafter stats -- is the "
                    f"server running with a drafter?"
                )
                failures += 1
                continue

            rate = (100.0 * accepted / drafted) if drafted else 0.0
            print(
                f"{want:>7} {t.get('prompt_n', 0):>9} {t.get('cache_n', 0):>7} "
                f"{drafted:>8} {accepted:>9} {rate:>7.1f}% "
                f"{t.get('draft_rounds', 0):>7} "
                f"{t.get('predicted_per_second', 0.0):>7.1f}"
            )

    print(
        "\n  Read it as a SLOPE, not as absolute numbers: the short prompt is the "
        "\n  unchunked control. Before patch 0032 the acceptance rate drops as soon "
        "\n  as the prompt crosses PREFILL_STEP; after it, the curve stays flat."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
