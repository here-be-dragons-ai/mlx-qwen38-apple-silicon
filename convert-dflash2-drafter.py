#!/usr/bin/env python3
"""Convert a DFlash-2 drafter from bf16 to MLX-quantized.

The official drafter (z-lab/Qwen3.8-27B-DFlash2) ships only in bf16 and occupies
3.58 GiB. Next to 15 GiB of model weights that is too much on 32 GB machines;
4bit brings it down to ~1.0 GiB.

Usage:
    ./convert-dflash2-drafter.py <source-bf16> <dest> [--bits 4] [--keep-codebooks]

    --bits 8            retains more quality (~1.9 GiB), in case acceptance
                        collapses at 4bit
    --keep-codebooks    leaves the two selector codebooks (2 x 248320 x 256) in
                        bf16. They are the lookup tables of the path selection
                        and react more sensitively to quantisation than the
                        projections do. Costs ~254 MiB and is the first knob to
                        turn when acceptance is poor.

Prerequisite: the DFlash-2 patch is applied (patches/apply-patches.sh),
otherwise mlx-vlm does not know the v2 modules.
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_vlm.speculative.drafters.qwen3_dflash.config import DFlashConfig
from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

CODEBOOKS = (
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 6, 8))
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--keep-codebooks", action="store_true")
    args = ap.parse_args()

    cfg_dict = json.loads((args.source / "config.json").read_text())
    config = DFlashConfig.from_dict(cfg_dict)
    if config.selector_rank == 0:
        raise SystemExit(
            "Source is not a DFlash-2 checkpoint (selector_rank missing from dflash_config)."
        )

    model = DFlashDraftModel(config)
    weights: dict = {}
    for f in sorted(args.source.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    model.load_weights(list(model.sanitize(weights).items()))

    before = sum(v.nbytes for _, v in tree_flatten(model.parameters()))

    def predicate(path: str, module) -> bool:
        if args.keep_codebooks and path in CODEBOOKS:
            return False
        if not hasattr(module, "to_quantized"):
            return False
        # Same rule as mlx-vlm on load (utils.py): weights not divisible by 64
        # stay unquantized, otherwise the kernel does not fit.
        return not (hasattr(module, "weight") and module.weight.size % args.group_size)

    nn.quantize(
        model, group_size=args.group_size, bits=args.bits, class_predicate=predicate
    )

    out = dict(tree_flatten(model.parameters()))
    after = sum(v.nbytes for v in out.values())

    args.dest.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.dest / "model.safetensors"), out, metadata={"format": "mlx"})

    # The quantisation block MUST go into config.json: on load, mlx-vlm
    # re-quantizes the freshly built model from this block and then checks per
    # module whether .scales are really present in the checkpoint
    # (utils.py:load_model).
    quant = {"group_size": args.group_size, "bits": args.bits}
    if args.keep_codebooks:
        for name in CODEBOOKS:
            quant[name] = False
    cfg_dict["quantization"] = quant
    cfg_dict["quantization_config"] = quant
    (args.dest / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    for extra in ("README.md",):
        src = args.source / extra
        if src.exists():
            shutil.copy2(src, args.dest / extra)

    gib = 1 << 30
    print(f"  Source    : {args.source}  ({before / gib:.2f} GiB of parameters)")
    print(f"  Dest      : {args.dest}  ({after / gib:.2f} GiB)")
    print(f"  Quant     : {args.bits} bit, group_size {args.group_size}"
          f"{', codebooks bf16' if args.keep_codebooks else ''}")
    print(f"  Tensors   : {len(out)}")


if __name__ == "__main__":
    main()
