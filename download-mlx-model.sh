#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace model downloader (curl, resumable)
#
# Downloads an MLX model into a FLAT directory (no HF cache layout with
# blobs/+snapshots/+refs/). Reasons:
#   - mlx-vlm/mlx-lm load from a normal path without trouble
#   - rebuilding the HF cache layout by hand (blob hashes, symlinks, refs) is
#     error-prone; a flat directory is more robust and can be inspected
#
# curl -C - resumes aborted downloads at the byte position. For files that are
# already complete, curl reports "already been transferred" (exit 33) -- treated
# as success here.
#
# Usage:
#   ./download-mlx-model.sh <hf-repo-id> <target-dir>
#   ./download-mlx-model.sh unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit ~/src/mlx/models/qwen36
#
# Running it again = resume. Aborting with Ctrl-C is safe at any time.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${1:?Usage: download-mlx-model.sh <hf-repo-id> <target-dir> [file-filter]}"
DEST="${2:?Usage: download-mlx-model.sh <hf-repo-id> <target-dir> [file-filter]}"
# Optional substring filter on the file name. MANDATORY for GGUF repos that
# carry several quantisations (bartowski/ggml-org/...) -- without a filter all
# variants would be downloaded there, i.e. hundreds of GB. Non-.gguf files
# (config, tokenizer, ...) are NOT excluded by the filter, so the model stays
# complete.
FILTER="${3:-}"

mkdir -p "$DEST"

echo "Repo   : $REPO"
echo "Target : $DEST"
[[ -n "$FILTER" ]] && echo "Filter : *${FILTER}* (matching .gguf only)"
echo "Fetching file list from HuggingFace..."

# .gitattributes/README are not needed (unsloth's README is ~65 KB of docs, but
# it does no harm -- deliberately included so the model card stays local).
FILES=$(curl -sL "https://huggingface.co/api/models/${REPO}" \
  | FILTER="$FILTER" python3 -c "
import sys, json, os
flt = os.environ.get('FILTER', '')
for f in json.load(sys.stdin).get('siblings', []):
    n = f['rfilename']
    if n == '.gitattributes':
        continue
    # The filter only applies to .gguf -- always take the metadata.
    if flt and n.endswith('.gguf') and flt not in n:
        continue
    print(n)")

[[ -n "$FILES" ]] || { echo "ERROR: no files found -- check the repo id: $REPO"; exit 1; }

# Fetch the expected sizes from the repo metadata. Without them a resume is
# blind: `curl -C -` starts at the local file size, and when the server returns
# a full 200 instead of a 206 (Partial Content) -- observed with HF after
# redirects and dropped connections -- curl appends the complete body TO THE END
# of the existing file. Result: an oversized, unusable file, WITHOUT curl
# reporting an error. That is exactly how shards of 5.18 instead of 4.77 GiB and
# 7.27 instead of 4.93 GiB appeared here (2026-08-11, after several restarts).
typeset -A SIZES
while IFS=$'\t' read -r _name _size; do
  [[ -n "$_name" ]] && SIZES[$_name]=$_size
done < <(curl -sL "https://huggingface.co/api/models/${REPO}?blobs=true" \
  | python3 -c "
import sys, json
for f in json.load(sys.stdin).get('siblings', []):
    s = f.get('size')
    if s:
        print(f\"{f['rfilename']}\t{s}\")")

fetch_one() {   # $1 = relative file name; returns: curl exit code
  local f="$1"
  local url="https://huggingface.co/${REPO}/resolve/main/${f}"
  local out="${DEST}/${f}"
  local exp="${SIZES[$f]:-0}"
  local have progress rc
  mkdir -p "$(dirname "$out")"

  if [[ -f "$out" && "$exp" -gt 0 ]]; then
    have=$(stat -f%z "$out" 2>/dev/null || echo 0)
    if [[ "$have" -eq "$exp" ]]; then
      echo "── $f  (complete, skipped)"
      return 0
    elif [[ "$have" -gt "$exp" ]]; then
      # Larger than expected = corrupted by a failed resume. Partial repair is
      # impossible because it is unclear at which offset the garbage starts --
      # so discard and fetch cleanly.
      echo "── $f  ⚠️  local $have > expected $exp bytes -- corrupted, re-downloading"
      rm -f "$out"
    fi
  fi

  echo "── $f"
  # -C -  : resume from the existing byte position
  # -L    : HF redirects to a CDN (cdn-lfs)
  # --retry: retry transient network/CDN errors automatically
  # Progress bar only on a terminal (tens of thousands of lines in a log).
  if [[ -t 1 ]]; then progress=(--progress-bar); else progress=(--no-progress-meter); fi

  set +e
  curl -L -C - --retry 5 --retry-delay 3 --retry-connrefused \
       "${progress[@]}" -o "$out" "$url"
  rc=$?
  set -e
  # Exit 33 = server does not support resume because the file is already complete.
  [[ $rc -eq 33 ]] && rc=0
  return $rc
}

echo "$FILES" | while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  exp="${SIZES[$f]:-0}"

  fetch_one "$f"; rc=$?

  # Verify the size. A "successful" curl run says NOTHING about whether the file
  # is correct (see the resume trap above) -- so verify hard.
  if [[ $rc -eq 0 && "$exp" -gt 0 ]]; then
    have=$(stat -f%z "${DEST}/${f}" 2>/dev/null || echo 0)
    if [[ "$have" -ne "$exp" ]]; then
      echo "   wrong size ($have instead of $exp) -- one clean retry from scratch"
      rm -f "${DEST}/${f}"
      fetch_one "$f"; rc=$?
      have=$(stat -f%z "${DEST}/${f}" 2>/dev/null || echo 0)
      [[ $rc -eq 0 && "$have" -ne "$exp" ]] && rc=1
    fi
  fi

  if [[ $rc -ne 0 ]]; then
    echo "ERROR: download failed ($f, exit $rc). Run the script again = resume."
    exit $rc
  fi
done

echo
echo "Done. Size: $(du -sh "$DEST" | cut -f1)"
echo "Contents:"
ls -la "$DEST"
