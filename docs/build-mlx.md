# Building mlx 0.32.2 from source

Optional. Everything runs without it -- only `head_dim 256` stays unfused, which
costs prefill headroom. The README links here from the installation section.

## Why

For Qwen3.8 the only real kernel win hangs on this: `head_dim 256` gets a fused
full-attention path again
([#4185](https://github.com/ml-explore/mlx/pull/4185), plus
[#3842](https://github.com/ml-explore/mlx/pull/3842) for NAX/M5). Both were
merged **after** the `0.32.1` tag, and **0.32.2 does not exist on PyPI to this
day** (not as `mlx-metal`/`mlx-cpu` either). That is why mlx is built here:

```sh
# Prerequisite: Xcode.app + Metal toolchain. The Command Line Tools are NOT
# enough — without them: xcrun: unable to find utility "metal"
sudo xcodebuild -license accept
xcodebuild -downloadComponent MetalToolchain      # ~690 MB
xcrun -sdk macosx metal --version                 # must succeed

git clone https://github.com/ml-explore/mlx.git ~/src/mlx-core
cd ~/src/mlx-core
CMAKE_GENERATOR=Ninja CMAKE_BUILD_PARALLEL_LEVEL=14 \
  uv build --wheel --python ~/src/mlx/.venv/bin/python

# mlx and mlx-metal share the mlx/ directory; the source build is monolithic,
# so remove BOTH first
uv pip uninstall --python ~/src/mlx/.venv/bin/python mlx mlx-metal
uv pip install   --python ~/src/mlx/.venv/bin/python dist/mlx-0.32.2.dev*.whl
```

> **`--python` is mandatory.** Without it `uv` builds against its default Python
> and drops a `cp313` wheel; the venv runs 3.12, and the wheel then does not fit.
>
> **`uv pip install -U mlx` falls back to 0.32.1** and disables patch `0013`
> again. The start banner shows it (`Full-Attn: unfused …`), but only after the
> next restart.

Patch `0013` is therefore live (`_FORCE_FUSED == True`). Measured with
`0.32.2.dev20260821+a082cb91`, production shape `qL=512` / `kL=22747`, 24 heads /
4 KV heads / `head_dim 256`:

| | peak |
|---|---|
| `force_fused=False` | 662 MiB |
| `force_fused=True` | **123 MiB** |

The difference of 539 MiB is the score transient (533 MiB in theory). The blowup
depends on the **asymmetric shape `qL << kL`**, not on the mask type — a
benchmark with `qL = kL` does *not* show it. Practical consequence on a 37.4 GiB
working set: the prefill previously died at 22,747 tokens with `[METAL]
Insufficient Memory`; afterwards **37,822 tokens went through in 91.8 s**.
