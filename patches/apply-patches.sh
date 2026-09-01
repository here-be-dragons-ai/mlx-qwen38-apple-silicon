#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm: apply local patches to site-packages.
#
# The patches live in site-packages and vanish on EVERY `uv pip install` /
# `pip install -U` of mlx-vlm. This script restores them. Idempotent (a reverse
# dry-run is used as the test).
#
#   ./apply-patches.sh            apply
#   ./apply-patches.sh --check    show status only
#   ./apply-patches.sh --revert   roll back
#
# venv Python via env:  MLX_VENV_PY=/path/to/.venv/bin/python ./apply-patches.sh
#
# STATE 2026-09-01: verified against mlx-vlm 0.7.0rc0 (tag 579cd51) and mlx
# 0.32.2. Eight patches, down from eleven. All eight apply to the tag unchanged
# -- no rebase was needed for the move off main @3fd38f4.
#
# Careful when moving past the tag: main has since drifted in models/base.py
# (0b73936, 9606b86, efd0479, all quantized-KV work) and patch 0013 no longer
# applies there -- one hunk of three, context drift only, mechanical to reanchor.
#
# The APC redesign (PR #1960, merged 2026-08-28) removed two of them:
#   0021  obsolete. _run_speculative is gone; non-MTP drafters no longer take a
#         separate loop that skips the APC manager, and apc_manager is wired
#         into the batching generator. MLX_VLM_SPECULATIVE_BATCH is referenced
#         nowhere upstream and the start script no longer sets it.
#   0030  replaced by a guard. Its cache-layer half landed upstream
#         (BatchQuantizedKVCache.is_trimmable/trim are now exactly what the patch
#         added); the verify-side half did not, and the verifier was rewritten.
#         The bug survives, and on 0.7.0rc0 it changed symptom again: no longer
#         a GPU Address Fault but HTTP 200 with corrupted text. #2113 (in this
#         tag) touches that rollback path and moved the failure mode without
#         fixing it. Isolated by elimination -- all three conditions are still
#         required. No profile ships MAX_NUM_SEQS>1, so the start script refuses
#         the combination instead of carrying a patch against rewritten code.
#         Upstream #1956/#1938 still open, neither in the tag.
#   0010  rebased onto the coordinator path. Still needed: three distinct
#         requests produced five snapshot files on main without it, three with.
#
# ── INCLUDED PATCHES ─────────────────────────────────────────────────────────
#
# 0010-qwen38-apc-single-snapshot.patch   (LOCAL, no upstream PR)
#   Suppresses the redundant full snapshot per request. Otherwise mlx-vlm stores
#   TWO nearly identical snapshots: the checkpoint at len-16 (guard) and the full
#   prompt. Measurements show the checkpoint is always the one that hits (prompt
#   3194 -> cached 3178, difference exactly 16); the full snapshot is dead weight
#   and halves the number of conversations kept warm.
#   MEASURED (APC_EXACT_CACHE_ENTRIES=2, M5 Pro):
#     without patch: "turn 1 repeated" cached=0     (full snapshot evicts everything)
#     with    patch: "turn 1 repeated" cached=3178, prefill 6737 -> 219 ms
#   Enabled via QWEN38_APC_SINGLE_SNAPSHOT=1 (set by the start script).
#   WITHOUT the env variable the patch is inert = exact upstream behaviour;
#   that is the rollback path.
#
# 0013-force-fused-sdpa-head-dim-256.patch   (LOCAL, no upstream PR)
#   STILL NEEDED on mlx 0.32.2 from PyPI: the kernels are there, but the default
#   dispatch still does not route to them -- only force_fused=True does.
#   Qwen3.8 has head_dim 256. mlx's default dispatch only permits fused full
#   attention for head_dim 64/80/128 -- the 16 full-attn layers therefore run on
#   the unfused graph and materialise a score transient of O(n_heads x qL x kL)
#   per layer. That is the actual reason PREFILL_STEP is a RAM lever here.
#   mlx 0.32.2 (PR #4185) restores the 192/256 kernels, reachable ONLY via
#   force_fused=True; the default dispatch still does not route there. The PR
#   justifies this explicitly by saying only the runtime knows its memory
#   budget -- which applies here.
#   Narrowly scoped: only qL > 1 (prefill/verify, not decode), only head_dim
#   192/256, only without an array mask and without sinks. If force_fused throws
#   once, the path is disabled permanently and logged once.
#   INERT ON mlx < 0.32.2: the import probe falls to TypeError.
#   VERIFIED on mlx 0.32.0: _FORCE_FUSED == False, behaviour unchanged.
#   Rollback: QWEN38_FORCE_FUSED_SDPA=0
#   CAREFUL: PR #3842 (fused head_dim 256 on NAX/M5) requires qL >= 1024 --
#   PROFILE=lean sets PREFILL_STEP=512 and therefore falls out of it.
#
# 0014-quantized-kv-start-uniform.patch   (LOCAL, no upstream PR)
#   quantized_kv_start applied on the batch path only for TurboQuant. On the
#   uniform path -- i.e. --kv-bits without --kv-quant-scheme turboquant, our
#   default -- quantisation happened from token 0, regardless of what
#   --quantized-kv-start said.
#   MEASURED with _make_cache(kv_bits=8, quantized_kv_start=8192):
#     without patch  prefill_length=1000  -> BatchQuantizedKVCache  (wrong)
#     with    patch  prefill_length=1000  -> BatchKVCache           (f16)
#                    prefill_length=20000 -> BatchQuantizedKVCache
#   AFFECTS PROFILE=lean in normal operation: KV_BITS=8 is the default there, and
#   since DFlash 2 became the default the start script sets
#   MLX_VLM_SPECULATIVE_BATCH=1 -- so the batch path no longer runs only at
#   MAX_NUM_SEQS > 1.
#   Rollback: QUANT_KV_START=0
#
# 0015-optional-fused-quantized-linears.patch   (LOCAL, no upstream PR)
#   _fused_quantized_linears() concatenates the QKV and MLP weights of each layer
#   into a fused tensor and attaches it to the module as
#   _qwen3_5_fused_decode_linears permanently -- a SECOND copy of the quantized
#   weights. Not a leak, an optimisation nobody releases.
#   This was the fixed memory floor: it appears on the FIRST generation, is
#   independent of context length (16 tokens trigger it just as much as 44,452)
#   and never comes back. Found with a probe around mx.eval:
#     8.71 GiB cumulative, n=128, language.py:1098 _target_verify_quantized_linears
#   MEASURED, idle after 5 requests on a 40 GiB working set:
#                        with fusion   without fusion   decode (mean of 5 each)
#     with spec decode     26.00 GiB       17.00 GiB     26.1 vs 25.7 tok/s
#     without spec decode  17.08 GiB       14.96 GiB     18.4 vs 18.2 tok/s
#   9 GiB against 1.5%, and the spread of the decode series overlaps completely.
#   The patch does NOT change behaviour by itself -- the default stays upstream.
#   The fusion is switched off by the start script via QWEN38_FUSED_LINEARS=0.
#   REBASED 2026-08-25: upstream renamed the function to
#   _decode_quantized_linears_fused in 0.6.16. The floor itself is NOT fixed
#   upstream -- _qwen3_5_fused_decode_linears is still attached to the module.
#   Rollback: QWEN38_FUSED_LINEARS=1
#
# 0021-speculative-apc-routing.patch   (LOCAL, upstream-PR candidate)
#   Makes the prefix cache reachable for non-MTP drafters at all.
#   server/generation.py routes every drafter except mtp into a second generation
#   loop (_run_speculative) that builds its own prompt cache and NEVER wires up
#   the apc_manager -- consequence: cached_tokens=0 on every request, and
#   APC_TRACE shows not a single lookup. The continuous-batching path has long
#   been able to do dflash (generic over draft_kind, receives apc_manager,
#   draft_kind and draft_block_size on the same line); only the switch kept it
#   away.
#   The patch makes the batch path reachable via MLX_VLM_SPECULATIVE_BATCH=1,
#   default unchanged. The start script sets the variable when DRAFT_KIND != mtp.
#   MEASURED (5.8k conversation, turn 2): cached 0 -> 5748/5788. Decode unchanged
#   (40.8 instead of 41.5 t/s on average), better on the 5767-token prompt
#   (38.4 -> 40.9 t/s). --draft-block-size still takes effect, MAX_NUM_SEQS=2
#   runs, MTP unchanged (cached 5772).
#   UPSTREAM STATUS: the corresponding issue #1966 was CLOSED on 2026-08-20 --
#   in favour of PR #1923 ("conservative DFlash APC prefix reuse", B=1 only,
#   text-only, exact-prefix). This patch will therefore not land in this form;
#   the dependency remains until #1923 is merged.
#
# 0041-dflash2-guard-invalid-bonus-token.patch   (LOCAL, no upstream PR)
#   Successor to 0022. Rebased on 2026-08-25 onto the upstream DFlash 2 from
#   PR #2014: the guard now sits in draft_block() of
#   speculative/drafters/dflash2/dflash2.py.
#   CAREFUL: speculative/drafters/qwen3_dflash/dflash.py still exists in 0.6.16
#   and is the v1 drafter. The patch applied cleanly there too -- and would have
#   been inert, guarding a path DFlash 2 no longer takes.
#   Content unchanged: for values outside the int64 range, mx.array() throws only
#   "RuntimeError: std::bad_cast", without the value, without an index
#   (reproducible with mx.array([[2**63]], dtype=mx.int32)). That is exactly how
#   a request died after 250 tokens on 2026-08-20 at 10:07. The guard checks
#   against vocab_size and names the value.
#   Deliberately no clamping: a silently replaced token corrupts the output.
#   CHECKED: PR #1959 does NOT have this guard -- the spot is open upstream.
#
# ── FOREIGN, STILL-OPEN UPSTREAM PRs (cherry-picked) ─────────────────────────
# Other people's bugfixes that are still open upstream. As soon as they are
# merged, this script reports "CONFLICT" -- remove them then.
#
# 0030-pr1956-speculative-quantized-kv.patch   (PR #1956, @Codcore, open)
#   "Fix speculative decoding against a quantized KV cache".
#   REPRODUCED HERE: with KV_BITS=8 and MAX_NUM_SEQS=2, two parallel requests die
#   with HTTP 500 and
#     AttributeError: 'tuple' object has no attribute 'shape'
#   The verify path assumes keys is ONE array; a quantized cache yields a tuple.
#   With the patch the same two requests run through correctly.
#   REBASED 2026-08-25: PR #2014 moved the verify path out of
#   models/qwen3_5/language.py into models/qwen3_5/speculative_verifier.py. The
#   bug moved with it -- speculative_verifier.py:1269 still does keys.shape[-2]
#   on something that is a tuple under a quantized cache. Both #1956 and #1938
#   are still open.
#   CLASSIFICATION CORRECTED ON 2026-08-20 -- this used to say the patch was only
#   relevant at MAX_NUM_SEQS > 1. That held while MTP was the default. Since
#   DFlash 2 became the default the start script sets MLX_VLM_SPECULATIVE_BATCH=1,
#   and _make_cache builds the batch cache even at MAX_NUM_SEQS=1 as soon as
#   KV_BITS is set (generate/ar.py:796). On PROFILE=lean, KV_BITS=8 is the
#   default -- so there this is NORMAL OPERATION, not a precaution.
#   TWO PRs FOR THE SAME THING: #1956 (here) and #1938 ("Fix Qwen speculative
#   decoding with quantized batch cache") change the same two files with the same
#   content. Only one will merge -- this patch covers both.
#
# 0031-pr1835-recurrent-cache-no-trim.patch    (PR #1835, @kylesyx, open)
#   "Decline prefix-cache reuse for non-trimmable recurrent caches".
#   _prefix_cache_trim_amount() only checks whether the prefix is still PRESENT,
#   not whether the cache is trimmable at all. The ArraysCache of Qwen3.8's 48
#   GDN layers is neither -- it passes the check and the caller dies on
#   c.trim(n_drop).
#   REPRODUCED HERE at unit level with the real cache classes:
#     without patch  _prefix_cache_trim_amount([ArraysCache, KVCache], 10) = 10
#                    -> AttributeError: 'ArraysCache' object has no attribute 'trim'
#     with    patch  = None (reuse declined); a pure KVCache model still returns
#                    10 -- no regression for attention models.
#   DOES NOT AFFECT OUR SERVER: _prefix_cache_trim_amount is only called from
#   dispatch.stream_generate, and the server path does not go through there.
#   Included as a precaution for mlx_vlm.chat_ui, the generate CLI and own
#   scripts that pass prompt_cache_state through.
#
# ── DONE / OBSOLETE ──────────────────────────────────────────────────────────
#
# 0040-pr1959-dflash2.patch   REMOVED 2026-08-25.
#   Carried upstream PR #1959 ("Add DFlash 2 speculative decoding"). That PR was
#   CLOSED UNMERGED on 2026-08-24 in favour of PR #2014, which landed in
#   mlx-vlm 0.6.16 and ships DFlash 2 at speculative/drafters/dflash2/.
#   The patch conflicts against 0.6.16 and is not needed -- the feature is
#   upstream. It lives in this repository's git history.
#   Note that speculative/drafters/qwen3_dflash/dflash.py still exists in 0.6.16
#   and is the v1 drafter; do not probe it to detect DFlash 2.

# 0020-dflash2-qwen38.patch   REPLACED 2026-08-20 by 0040 (upstream PR #1959).
#   The own transcription from z-lab/dflash was correct -- including the codebook
#   rename that z-lab itself only canonicalised on 2026-08-18 with e128a7e and
#   that #1959 makes identical. It was replaced anyway: #1959 adds the exact
#   4bit M=4 verifier kernel and distribution-preserving rejection sampling.
#   It lives in this repository's git history.
#
# 0022-dflash-guard-invalid-bonus-token.patch   REPLACED 2026-08-20 by 0041.
#   Same guard, different place: since #1959, DFlash 2 builds the block in
#   DFlash2DraftModel.propose_block instead of DFlashDraftModel.draft_block.
#
# 0002-pr1901-apc-short-prompt.patch   REMOVED 2026-08-19 when upgrading to
#   mlx-vlm 0.6.15. Upstream PR #1901 was merged on 2026-08-15, i.e. three days
#   AFTER the 0.6.13 release -- which is why it was still needed in 0.6.13 and is
#   included from 0.6.14. Cross-checked by reverse dry-run against the 0.6.15
#   wheel: both hunks (apc.py, server/app.py) are in.
#   What it fixed: a prompt shorter than APC_EXACT_PREFIX_GUARD_TOKENS (16) made
#   the checkpoint length negative; clamped to 1, a ONE-TOKEN snapshot was stored
#   that matched the beginning of every later prompt. Since a new checkpoint is
#   only created when nothing was reused, that tenant never stored a usable one
#   again.
#   Signature in the log: cached_tokens=1 on large prompts.
#   MEASURED (M5 Pro, 48 GB): 13 requests with cached_tokens==1 = 1055 s of lost
#   prefill time, triggered by only 6 short prompts.
#   ANYONE GOING BACK TO 0.6.13 needs it again -- it lives in this repository's
#   git history.
#
# DELIBERATELY NOT INCLUDED:
#   The KV window patch (QWEN38_KV_WINDOW, a sliding window over the 16 full-attn
#   layers) from the 48 GB machine. It halves exactly the quantity that hurts on
#   32 GB, but was DISPROVED on 2026-08-17 with needle_hybrid.py: a needle
#   outside the window was lost and in the 40% case even hallucinated (8347
#   instead of 8342). Save memory here via KV_BITS=8 and a smaller context, not
#   via windowing.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PATCH_DIR="${0:A:h}"
VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"

[[ -x "$VENV_PY" ]] || { echo "ERROR: venv Python not found: $VENV_PY"; exit 1; }
SITE_PACKAGES=$("$VENV_PY" -c "import mlx_vlm, os; print(os.path.dirname(os.path.dirname(mlx_vlm.__file__)))" 2>/dev/null) \
  || { echo "ERROR: mlx_vlm is not importable in $VENV_PY."; exit 1; }

MODE="apply"
[[ "${1:-}" == "--revert" ]] && MODE="revert"
[[ "${1:-}" == "--check" ]]  && MODE="check"

echo "  site-packages: $SITE_PACKAGES"
echo "  mlx-vlm      : $("$VENV_PY" -c 'import importlib.metadata as m;print(m.version("mlx-vlm"))')"

patches=( "$PATCH_DIR"/*.patch(N) )
[[ ${#patches[@]} -gt 0 ]] || { echo "  No .patch files in $PATCH_DIR"; exit 0; }

# Reverting happens in REVERSE order. That mattered while 0041 sat on code that
# 0040 created: reverting 0040 first no longer found the expected context, the
# reverse dry-run failed, and the patch stayed in silently. 0040 is gone since
# 2026-08-25, so no patch currently depends on another -- the reverse order and
# the copy-based probe below are kept because the next stacked patch would
# reintroduce exactly that failure, and it fails quietly.
# (${(Oa)...} reverses.)
# IMPORTANT: revert_order stays a separate variable. Reversing `patches` itself
# would make the probe below reverse it a second time -- it would then check in
# apply instead of teardown order.
revert_order=( ${(Oa)patches} )

# ── Determine applied status ─────────────────────────────────────────────────
# The reverse dry-run alone is no longer enough now that patches sit on top of
# each other: 0041 lives IN THE MIDDLE of the code that 0040 creates. A reverse
# dry-run of 0040 then does not find its own context again and wrongly reports it
# as open -- whereupon a second run would try to apply it again and shout
# "CONFLICT".
# The status is therefore determined on a COPY that is torn down in reverse:
# 0041 first, after which 0040 stands free again. 15 MB, once per run.
_probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlxvlm-probe.XXXXXX")
trap 'rm -rf "$_probe_dir"' EXIT INT TERM
cp -R "$SITE_PACKAGES/mlx_vlm" "$_probe_dir/"
typeset -A applied_map
for p in "${revert_order[@]}"; do
  if patch -R -p1 --dry-run --force -d "$_probe_dir" < "$p" &>/dev/null; then
    applied_map[${p:t}]=1
    patch -R -p1 --force -d "$_probe_dir" < "$p" &>/dev/null || true
  else
    applied_map[${p:t}]=0
  fi
done
rm -rf "$_probe_dir"
trap - EXIT INT TERM

[[ "$MODE" == "revert" ]] && patches=( "${revert_order[@]}" )

for p in "${patches[@]}"; do
  name="${p:t}"
  applied=${applied_map[$name]}

  case "$MODE" in
    check)
      echo "  $([[ $applied == 1 ]] && echo '[ applied ]' || echo '[  open   ]')  $name"
      continue
      ;;
    revert)
      if [[ $applied == 1 ]]; then
        patch -R -p1 -d "$SITE_PACKAGES" < "$p" >/dev/null
        echo "  reverted: $name"
      else
        echo "  not applied, skipped: $name"
      fi
      continue
      ;;
  esac

  if [[ $applied == 1 ]]; then
    echo "  already applied: $name"
    continue
  fi

  if ! patch -p1 --dry-run -d "$SITE_PACKAGES" < "$p" &>/dev/null; then
    echo "  ⚠️  CONFLICT: $name cannot be applied."
    echo "      Usually means: the target file changed upstream."
    echo "      Check whether the effect has landed upstream in the meantime --"
    echo "      if so, remove the file from $PATCH_DIR, otherwise rewrite the patch."
    continue
  fi

  patch -p1 -d "$SITE_PACKAGES" < "$p" >/dev/null
  echo "  applied: $name"
done

[[ "$MODE" == "check" ]] && exit 0

# A patch that only partially applies against a changed upstream version shows
# up here.
if ! "$VENV_PY" -c "import mlx_vlm.apc, mlx_vlm.generate.ar" 2>/dev/null; then
  echo "  ⚠️  WARNING: mlx_vlm is NOT importable after patching."
  echo "      Roll back with:  $0 --revert"
  exit 1
fi
echo "  mlx_vlm importable -- ok."
