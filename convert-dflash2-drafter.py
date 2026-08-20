#!/usr/bin/env python3
"""DFlash-2-Drafter von bf16 nach MLX-quantisiert konvertieren.

Der offizielle Drafter (z-lab/Qwen3.8-27B-DFlash2) liegt nur in bf16 vor und
belegt 3,58 GiB. Neben 15 GiB Modellgewichten ist das auf 32-GB-Maschinen zu
viel; 4bit bringt ihn auf ~1,0 GiB.

Verwendung:
    ./convert-dflash2-drafter.py <quelle-bf16> <ziel> [--bits 4] [--keep-codebooks]

    --bits 8            haelt mehr Qualitaet (~1,9 GiB), falls die Acceptance
                        bei 4bit einbricht
    --keep-codebooks    laesst die beiden Selector-Codebooks
                        (2 x 248320 x 256) in bf16. Sie sind die
                        Nachschlagetabellen der Pfadwahl und reagieren
                        empfindlicher auf Quantisierung als die Projektionen.
                        Kostet ~254 MiB, ist die erste Stellschraube bei
                        schlechter Acceptance.

Voraussetzung: der DFlash-2-Patch ist angewendet (patches/apply-patches.sh),
sonst kennt mlx-vlm die v2-Module nicht.
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
            "Quelle ist kein DFlash-2-Checkpoint (selector_rank fehlt in dflash_config)."
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
        # Gleiche Regel wie mlx-vlm beim Laden (utils.py): nicht durch 64
        # teilbare Gewichte bleiben unquantisiert, sonst passt der Kernel nicht.
        return not (hasattr(module, "weight") and module.weight.size % args.group_size)

    nn.quantize(
        model, group_size=args.group_size, bits=args.bits, class_predicate=predicate
    )

    out = dict(tree_flatten(model.parameters()))
    after = sum(v.nbytes for v in out.values())

    args.dest.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.dest / "model.safetensors"), out, metadata={"format": "mlx"})

    # Die Quantisierungsangabe MUSS in die config.json: mlx-vlm quantisiert das
    # frisch gebaute Modell beim Laden anhand dieses Blocks nach und prueft dann
    # je Modul, ob im Checkpoint wirklich .scales liegen (utils.py:load_model).
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
    print(f"  Quelle    : {args.source}  ({before / gib:.2f} GiB Parameter)")
    print(f"  Ziel      : {args.dest}  ({after / gib:.2f} GiB)")
    print(f"  Quant     : {args.bits} bit, group_size {args.group_size}"
          f"{', Codebooks bf16' if args.keep_codebooks else ''}")
    print(f"  Tensoren  : {len(out)}")


if __name__ == "__main__":
    main()
