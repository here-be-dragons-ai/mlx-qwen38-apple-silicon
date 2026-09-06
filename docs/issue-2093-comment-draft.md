<!-- posted 2026-09-06 to https://github.com/Blaizzy/mlx-vlm/issues/2093
     comment: https://github.com/Blaizzy/mlx-vlm/issues/2093#issuecomment-5561642675
     author: gtonic (the account authenticated in `gh`)
     line numbers cite tag v0.7.0rc0, not this working tree -->

Independent confirmation with a different drafter and different hardware, plus one detail that may matter for the fix order.

## Setup

- Apple M5 Pro, 48 GB, macOS 26, `iogpu.wired_limit_mb=40960`
- mlx-vlm 0.7.0rc0 (tag `v0.7.0rc0`), mlx 0.32.2, mlx-lm 0.31.3
- `mlx-community/Qwen3.8-27B-4bit` (`qwen3_5`, hybrid GDN + full attention, 64 layers, 16 of them full-attention)
- Server, continuous batching, one sequence, `--kv-bits 8 --quantized-kv-start 8192`
- Drafter: **DFlash 2** (`draft_kind="dflash"`), `block_size 4` — not MTP

## Defect 1 reproduces, and it is not MTP-specific

The branch that reaches `make_speculative_prompt_cache` is keyed on a drafter existing at all, not on its kind (`mlx_vlm/generate/ar.py:1762` at the tag):

```python
elif draft_model is not None and draft_kind is not None:
    self.prompt_cache = make_speculative_prompt_cache(
        model,
        draft_kind=draft_kind,
        batch_size=len(input_ids),
        left_padding=left_padding,
        make_cache=lambda lm, lp: _make_cache(lm, lp, kv_bits=kv_bits, ...),
    )
```

and `mlx_vlm/speculative/utils.py:125` discards that factory:

```python
if batch_size == 1:
    return cache.make_prompt_cache(lm)
```

What makes this easy to miss from the outside is the *next* branch, a few lines below at `generate/ar.py:1785`, which guards the dense path explicitly:

```python
elif (
    len(input_ids) == 1
    and right_pad_per_row is None
    and kv_bits is None
    and hasattr(model, "make_cache")
):
    self.prompt_cache = cache.make_prompt_cache(model)
```

So without a drafter `kv_bits` is honoured, and with any drafter at `batch_size == 1` it is silently not — adjacent branches in the same file, opposite behaviour. Any single-stream server with a drafter is affected, which is the common serving shape.

## Confirmed on disk, independently of reading the code

Safetensors header of an exact-APC snapshot written by that server:

```
c3_kind: 'kv'   c7_kind: 'kv'   c11_kind: 'kv'  ...  c63_kind: 'kv'
dtypes: {'BF16': 80, 'F32': 48}
```

All 16 full-attention layers stored as dense `kind: "kv"`, and no packed integer tensors anywhere in the file. That matches your observation that such a server writes dense snapshots — here from the `dflash` path rather than MTP.

## One consideration for the fix order

Once the cache genuinely is quantized, attention takes the `hasattr(cache, "bits")` branch at `mlx_vlm/models/base.py:397` and goes to `quantized_scaled_dot_product_attention`. That path never reaches `mx.fast.scaled_dot_product_attention`, so for `head_dim` 256 it also never reaches the fused kernels.

#2163 reports what that path then does at long context — the full L×S score matrix in one allocation, a higher peak than f16, and `kIOGPUCommandBufferCallbackErrorOutOfMemory` — on the same model and the same mlx/mlx-vlm versions. Fixing this issue on its own would route single-stream users from "quantization silently does nothing" into that. The two may want to land together, or at least reference each other.

Happy to run further checks on this configuration.
