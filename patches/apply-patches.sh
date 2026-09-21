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
# STATE 2026-09-21: verified against mlx-vlm 0.7.2 (tagged at a74c7de) and
# mlx 0.32.2. SEVEN patches, unchanged from the 09-17 set.
#   - Install: uv pip install "mlx-vlm==0.7.2"  (no --no-deps any more; 0.7.2
#     declares mlx>=0.32.2, a lower bound, so the exact mlx pin survives the
#     same resolution. Under the previous GIT pin it did not, and patch 0013
#     fell inert when uv dropped mlx to 0.32.1.)
#   - All seven applied to a74c7de without fuzz. Expected: the tag's apc.py,
#     apc_adapters.py, apc_coordinator.py, models/base.py, speculative/ and
#     server/generation.py are byte-identical to main @ 548b09b, which is what
#     they were verified against on 09-17.
#   - NEVER 0.7.1. That tag carries #2182 without its fix #2262, and #2259 is
#     the consequence: on a hybrid model a SHORT prompt permanently inflates the
#     prefill reserve, after which exact APC silently stores and restores
#     nothing for the rest of the process lifetime. Our traffic is short agent
#     turns. 0.7.2 is the first release without that defect.
#   - 0033 is GONE, superseded by #2262/#2182 -- see the DONE section.
#   - 0015 was REANCHORED on 09-17 into a different file; upstream moved the
#     function. Still there in 0.7.2.
#
# WHAT #2262 CHANGED FOR US (mlx-vlm PR, merged 2026-09-16). APC sized the
# prefill reserve from the largest snapshot-bytes/token ratio the process had
# ever seen, as a monotonic max. Short checkpoints carry the fixed GDN recurrent
# state and unused KV capacity, so one 37-token request set a ratio ~10x too
# high and every later prompt over-reserved. Gone with the PR:
# _bytes_per_token, _cache_size_estimate and both proportional exact-restore
# estimates; planning now runs through the cache adapters with real tensor
# dimensions. VERIFIED HERE: those three symbols no longer exist in apc.py.
#
# That warning about models/base.py drift is retired: the release carries the
# quantized-KV work and #1822, and patch 0013 applied to it without a change.
# Re-verified on 0.7.0 through the patched entry point: 181 MiB at qL=2048 /
# kL=22747, against ~2362 MiB unfused.
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
#   192/256, only without sinks.
#
#   REWRITTEN 2026-09-02, AND IT ONLY STARTED WORKING ON THE SERVER THEN.
#   The old condition also required "no array mask", and that excluded exactly
#   the production case: the server runs every request through the batching
#   generator, whose BatchKVCache.make_mask() ALWAYS returns an array (it
#   encodes left_padding) and never the "causal" string. The patch was applied,
#   the start banner said "fused", and all 16 layers ran unfused. force_fused
#   handles array masks perfectly well -- the restriction was unfounded.
#   MEASURED through the patched entry point with the real server mask,
#   24 q-heads / 4 kv-heads / head_dim 256, bf16, per layer:
#     qL=2048 / kL=22747   unfused 2362 MiB, 83.9 ms -> fused 205 MiB, 68.7 ms
#     qL= 512 / kL=22747   unfused  675 MiB, 20.6 ms -> fused 136 MiB, 17.2 ms
#   Numerically the fused kernel is the better one (max error against an fp32
#   reference 0.0011 vs 0.0050): it accumulates in fp32 instead of materialising
#   the scores in fp16. Verified for a left-padded B=2 batch as well.
#   The start script's FUSED_OK probe was rewritten with it -- it used to ask
#   only whether mlx knows the force_fused argument, which said nothing about
#   whether the path is taken.
#
#   THE REFUSAL SET, AND WHY IT IS KEYED BY SHAPE: there is a narrow hole in the
#   kernel coverage, measured on mlx 0.32.2 at GQA factor 6 / head_dim 256 --
#     qL <= 5 fused (sdpa_vector wants qL x GQA <= 32) / qL 6,7,8 NO KERNEL /
#     qL >= 12 fused
#   -- and it sits where speculative verify runs. Until 2026-09-02 a single
#   throw set a global flag and the fused path was gone for the whole process,
#   including the qL=2048 prefills that carry the 2.1 GiB per layer.
#   _FORCE_FUSED_REFUSED is now keyed by (head_dim, dtype, mask kind, qL): one
#   refused shape disables one shape. DRAFT_BLOCK_SIZE=4 (the default) verifies
#   at qL 4-5 and stays clear of the hole; 7 or 8 lands in it.
#   INERT ON mlx < 0.32.2: the import probe falls to TypeError.
#   VERIFIED on mlx 0.32.0: _FORCE_FUSED == False, behaviour unchanged.
#   Rollback: QWEN38_FORCE_FUSED_SDPA=0
#   ON PR #3842 (fused head_dim 256 on NAX/M5, qL >= 1024): the win does NOT
#   depend on it. At qL=512 the fused path engages and saves 539 MiB per layer,
#   so PROFILE=lean is not cut off from it.
#   UPSTREAM: mlx PR #4416 (merged 2026-09-02, unreleased) routes head_dim 256
#   with an array mask through the DEFAULT dispatch. Once that is in a release,
#   the array-mask half of this patch is redundant; the force_fused half is not.
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
#   UNTIL 2026-09-07 THIS PATCH COULD NOT FIRE ON A SERVER WITH A DRAFTER, and
#   neither could --kv-bits itself. make_speculative_prompt_cache returned the
#   model's plain dense cache at batch_size == 1 and dropped the _make_cache
#   factory that carries kv_bits -- upstream issue #2093, which we confirmed
#   here from the safetensors headers of our own APC snapshots (all 16
#   full-attention layers stored dense while the banner advertised 32 KiB/token).
#   mlx-vlm 0.7.0 ships #1822, which removes that bypass: every drafter now
#   builds its prompt cache through make_cache. So this patch, and KV
#   quantisation in general, became reachable on this setup with that release.
#   STILL UNMEASURED, and deliberately so: once quantisation actually engages,
#   attention takes the hasattr(cache, "bits") branch and never reaches the
#   fused kernels of patch 0013. Upstream #2163 reports what that path then does
#   at long context -- the full L x S score matrix in one allocation, a higher
#   peak than f16 below ~50k, and OOM at 200k -- on this exact model and these
#   versions. Measure before enabling; do not set it as a profile default.
#
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
#   _decode_quantized_linears_fused in 0.6.16.
#   REANCHORED 2026-09-17 INTO A DIFFERENT FILE. 0.7.1 lifted the function out of
#   models/qwen3_5/language.py into the shared speculative/ops/linear.py, and the
#   module attribute lost its prefix with it (_qwen3_5_fused_decode_linears ->
#   _fused_decode_linears). The body is otherwise unchanged, so the patch is the
#   same two lines in a new place. Two consequences worth knowing:
#     - The switch is no longer qwen3_5-specific. Every model routed through
#       _target_verify_linears now sees it. That is harmless because it is opt-in
#       and off by default upstream, but it is no longer a Qwen-local lever.
#     - The old anchor is the dangerous part. Against models/qwen3_5/language.py
#       hunk 1 (import os) still applies while hunk 2 does not, so a --forward
#       apply outside this script leaves a tree that looks patched and has an
#       INERT switch -- the exact failure mode patch 0013 sat in for six weeks.
#       This script dry-runs the whole patch first and reports CONFLICT instead,
#       and the start script greps the NEW path for the marker.
#   The floor itself is still NOT fixed upstream: there is no opt-out in
#   speculative/ops/linear.py, checked on main @ 548b09b.
#   VERIFIED 2026-09-17 through the real entry point: with fusable 4bit linears
#   the default path fuses and attaches _fused_decode_linears; with
#   QWEN38_FUSED_LINEARS=0 it returns None, attaches nothing, and the outputs of
#   _target_verify_linears are bit-identical (max abs diff 0.0).
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
# ── FOREIGN UPSTREAM PRs (cherry-picked) ─────────────────────────────────────
# Other people's bugfixes that are still open upstream. As soon as they are
# merged, this script reports "CONFLICT" -- remove them then, which is exactly
# what happened to 0031 and 0034 with the 0.7.0 release, and to 0033 on
# 2026-09-17.
# NONE are left. Since 0033 went, this patch set is entirely local work: seven
# patches, no upstream PR among them, each one a lever this machine needs and
# upstream has no reason to ship.
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
#   BROADER THAN THIS PATCH (2026-09-11): issue #2033 got an independent
#   reproduction on the 0.7.0 release -- MTP drafter on a dense qwen3_5 6-bit
#   target, --max-num-seqs 8, f16 KV, NO --kv-bits anywhere. At concurrency 2 the
#   output degenerates into token_id=0 ('!') and comes back as HTTP 200 with
#   corrupt content, non-streaming included; the no-drafter control is correct at
#   concurrency 1, 2 and 4. So drafter plus concurrency corrupts on its own,
#   without the quantized cache this patch is about, and nothing above INFO is
#   logged -- a caller cannot detect it. PR #2197 was tested there and does NOT
#   fix it. That is the strongest argument yet for the start script refusing
#   MAX_NUM_SEQS > 1 outright instead of carrying a patch for it.
#
# ── CONSIDERED AND DECLINED ──────────────────────────────────────────────────
#
# 0035-issue2210-apc-single-row-plain  DECLINED 2026-09-10, before it was ever
#   written. Upstream issue #2210 (@felix-ab) ships a 25-line fix as a gist:
#   when the exact-APC hit restores exactly one row and no KV quantisation is
#   configured, return the plain single-row clone instead of merging it into
#   batch-aware caches, which stops the per-decode-token copy of the whole KV +
#   recurrent state.
#   MEASURED HERE on 0.7.0 (./measure-apc-warm-decode.py, 28,590 tokens, 300
#   decoded, temperature 0, PROFILE=roomy):
#     drafter on   warm/cold decode 1.003 (3 pairs) and 1.004 (2 pairs)
#     drafter off  warm/cold decode 0.645 (2 pairs) and 0.628 (2 pairs)
#   The second column of each row reverts 0010 and 0033 (0033 was still carried
#   then), i.e. the defect is upstream's and not an artefact of our own APC
#   patches. It is REAL on 0.7.0 --
#   and invisible on this server, because every profile runs a drafter and the
#   speculative path does not take the shortcut. A patch against a code path we
#   never execute is the same bad trade that removed 0032.
#   REVISIT IF: a profile ever ships ENABLE_SPEC_DECODE=0, or a release changes
#   which decode path the drafter takes. See docs/memory.md and
#   docs/issue-2210-comment-draft.md.
#
# ── DONE / OBSOLETE ──────────────────────────────────────────────────────────
#
# 0033-pr2072-apc-ownership-transfer-peaks.patch   REMOVED 2026-09-17 with the
#   move to main @ 548b09b. Carried PR #2072 ("Reduce exact APC ownership-
#   transfer peaks"), added 2026-09-04 against a measured failure: the exact-APC
#   snapshot store cloned the live prompt cache and evaluated the copy, a second
#   1.97 GiB of KV at a 32,256-token prompt, asked for while the working set
#   stood at 95%. Three OOMs in two days sat at that call site -- the trace is in
#   the APC_ENTRIES block of start-mlx_qwen3.8.sh.
#   #2072 itself is STILL OPEN upstream. It was replaced anyway, by #2182 and
#   #2262, which attack the same peak from the other end: the manager now sizes
#   a snapshot before deciding, keeps at most memory_max_bytes resident, and
#   spills anything larger straight to disk -- explicitly "without a second full
#   snapshot just to spill", which is the sentence #2072 was carried for.
#   THE EVIDENCE IS IN THE REJECTS, and it is why this is a deletion and not a
#   reanchor:
#     against the 0.7.1 tag   2 of 13 hunks fail
#     against main @ 548b09b  5 of 13 hunks fail, across all four files
#       (apc.py 1/5, apc_adapters.py 2/3, apc_coordinator.py 1/2, ar.py 1/3)
#   The ar.py hunk failed because upstream now contains it VERBATIM: 0.7.1 calls
#   coordinator.store_checkpoint(self.prompt_cache, batch_idx=...) and lets
#   snapshot_prompt_cache_row(..., clone=False) do the extraction, which is
#   exactly what the patch rewrote that call site into. The rest of #2072 was
#   overwritten by the #2262 planner rewrite. Reanchoring five hunks onto code
#   that is being rewritten weekly, for a peak the rewrite already bounds, is
#   the wrong trade.
#   WHAT IS LOST, honestly: #2072's per-layer requantise-on-restore and the
#   no-copy promotion of a consumed single KVCache row are NOT upstream. If the
#   restore path ever shows a peak again, that is where to look first.
#   THE OPEN QUESTION IT LEAVES BEHIND is unchanged and still unanswered: the
#   immediate lever against those OOMs was APC_ENTRIES 3 -> 2, and whether the
#   3 can come back is now a question about the #2182 resident ceiling
#   (4.0 GiB here) rather than about this patch. Measure before raising it.
#   It lives in this repository's git history.
#
# 0031-pr1835-recurrent-cache-no-trim.patch   REMOVED 2026-09-07 with the move
#   to mlx-vlm 0.7.0. Carried PR #1835 (@kylesyx), which upstream CLOSED UNMERGED
#   on 2026-09-03 in favour of #2152 -- a narrower one-line fix in the same
#   function, which the release contains: dispatch.py now reads
#   `c.is_trimmable() and _cache_fully_retained(c)`.
#   Note what was lost with the broader PR: #2152 still sits behind `if n_drop`,
#   so a bare ArraysCache or a CacheList with no top-level offset collapses
#   cached_len to 0, skips the guard and is accepted for reuse. kylesyx had
#   covered that; upstream took the narrow fix. Not our server path, but the
#   hole is now unowned.
#
# 0034-pr2090-packed-apc-checkpoints.patch   REMOVED 2026-09-07 with the move to
#   mlx-vlm 0.7.0, which contains PR #2090. It lived exactly five days and never
#   ran in production: it landed on 09-04, and the server that would have used it
#   was started before it. Its whole point -- exact-APC snapshots staying packed
#   instead of being dequantized on store -- is now the release behaviour.
#
# 0032-pr2096-chunked-prefill-drafter-priming.patch   REMOVED 2026-09-04,
#   one day after it was added. Carried upstream PR #2096 ("Prime speculative
#   drafters from the whole prompt during chunked prefill"), which is still open.
#   It works, and it was still the wrong trade for this machine.
#   MEASURED 2026-09-03 (n=12 per arm, one server restart each, 400 generated
#   tokens, temperature 0): acceptance 48.2% -> 54.7%, +6.5 pp pooled, Welch
#   t = 6.34, and target passes per 400 tokens down about 6%. Decode throughput
#   did NOT move: -3.2% / +1.0% / +1.2% across 1024/4096/8192, inside the noise.
#   The unchunked 1024 control improved too (+5.3 pp), which upstream's framing
#   does not predict -- the patch also routes speculative_prompt_ids differently
#   and that applies to every prompt.
#   WHAT KILLED IT was the memory side, which was never measured before it went
#   in. DFlash captures five target layers (5, 19, 33, 47, 61) at hidden_size
#   5120, i.e. 50 KiB per token across them in bf16. Without the patch only the
#   final chunk is captured (~100 MiB). With it, every chunk's capture is held
#   and then concatenated by splice_prompt_hidden:
#     32,256 tokens  ~1.5 GiB held, ~3 GiB through the concatenation
#     65,536 tokens  ~3.1 GiB held, ~6 GiB
#   On a machine running at 95% of its working set that is decisive, and the OOM
#   on 2026-09-03 21:39:18 sits exactly at its mx.eval(chunk_hidden) in
#   prompt_step(). +6.5 pp acceptance, no throughput, 1.5-3 GiB of peak: a bad
#   trade here. Reopen it when memory is not the binding constraint.
#   It lives in this repository's git history.
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
