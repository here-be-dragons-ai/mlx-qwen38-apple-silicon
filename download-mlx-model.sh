#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace-Modell-Downloader (curl, resume-fähig)
#
# Laedt ein MLX-Modell in ein FLACHES Verzeichnis (kein HF-Cache-Layout mit
# blobs/+snapshots/+refs/). Gruende:
#   - mlx-vlm/mlx-lm laden problemlos von einem normalen Pfad
#   - das HF-Cache-Layout von Hand nachzubauen (Blob-Hashes, Symlinks, refs)
#     ist fehleranfaellig; ein flaches Verzeichnis ist robuster und inspizierbar
#
# curl -C - setzt abgebrochene Downloads an der Byte-Position fort. Bei bereits
# vollstaendigen Dateien meldet curl "already been transferred" (Exit 33) — das
# wird hier als Erfolg behandelt.
#
# Verwendung:
#   ./download-mlx-model.sh <hf-repo-id> <zielverzeichnis>
#   ./download-mlx-model.sh unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit ~/src/mlx/models/qwen36
#
# Erneut ausfuehren = Resume. Abbruch mit Ctrl-C ist jederzeit gefahrlos.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${1:?Usage: download-mlx-model.sh <hf-repo-id> <target-dir> [datei-filter]}"
DEST="${2:?Usage: download-mlx-model.sh <hf-repo-id> <target-dir> [datei-filter]}"
# Optionaler Substring-Filter auf den Dateinamen. PFLICHT bei GGUF-Repos, die
# mehrere Quantisierungen fuehren (bartowski/ggml-org/...) — ohne Filter wuerden
# dort alle Varianten geladen, also hunderte GB. Nicht-.gguf-Dateien (config,
# tokenizer, ...) werden vom Filter NICHT ausgeschlossen, damit das Modell
# vollstaendig bleibt.
FILTER="${3:-}"

mkdir -p "$DEST"

echo "Repo   : $REPO"
echo "Ziel   : $DEST"
[[ -n "$FILTER" ]] && echo "Filter : *${FILTER}* (nur passende .gguf)"
echo "Dateiliste von HuggingFace holen..."

# .gitattributes/README werden nicht gebraucht (README ist bei unsloth ~65 KB Doku,
# schadet aber nicht — hier bewusst mitgenommen, damit die Modellkarte lokal bleibt).
FILES=$(curl -sL "https://huggingface.co/api/models/${REPO}" \
  | FILTER="$FILTER" python3 -c "
import sys, json, os
flt = os.environ.get('FILTER', '')
for f in json.load(sys.stdin).get('siblings', []):
    n = f['rfilename']
    if n == '.gitattributes':
        continue
    # Filter greift nur auf .gguf — Metadaten immer mitnehmen.
    if flt and n.endswith('.gguf') and flt not in n:
        continue
    print(n)")

[[ -n "$FILES" ]] || { echo "ERROR: Keine Dateien gefunden — Repo-ID pruefen: $REPO"; exit 1; }

# Erwartete Groessen aus den Repo-Metadaten holen. Ohne sie ist ein Resume blind:
# `curl -C -` setzt an der lokalen Dateigroesse an, und wenn der Server statt
# eines 206 (Partial Content) ein volles 200 liefert — bei HF nach Redirects und
# abgerissenen Verbindungen beobachtet — haengt curl den kompletten Body ANS ENDE
# der vorhandenen Datei. Ergebnis: eine zu grosse, unbrauchbare Datei, OHNE dass
# curl einen Fehler meldet. Genau so sind hier Shards mit 5,18 statt 4,77 GiB und
# 7,27 statt 4,93 GiB entstanden (2026-08-11, nach mehreren Neustarts).
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

fetch_one() {   # $1 = relativer Dateiname; Rueckgabe: curl-Exitcode
  local f="$1"
  local url="https://huggingface.co/${REPO}/resolve/main/${f}"
  local out="${DEST}/${f}"
  local exp="${SIZES[$f]:-0}"
  local have progress rc
  mkdir -p "$(dirname "$out")"

  if [[ -f "$out" && "$exp" -gt 0 ]]; then
    have=$(stat -f%z "$out" 2>/dev/null || echo 0)
    if [[ "$have" -eq "$exp" ]]; then
      echo "── $f  (vollstaendig, uebersprungen)"
      return 0
    elif [[ "$have" -gt "$exp" ]]; then
      # Groesser als erwartet = durch einen fehlgeschlagenen Resume beschaedigt.
      # Teilweise reparieren geht nicht, weil unklar ist, ab welchem Offset der
      # Muell beginnt — also verwerfen und sauber neu holen.
      echo "── $f  ⚠️  lokal $have > erwartet $exp Bytes — beschaedigt, laedt neu"
      rm -f "$out"
    fi
  fi

  echo "── $f"
  # -C -  : Resume ab vorhandener Byte-Position
  # -L    : HF leitet auf CDN (cdn-lfs) um
  # --retry: transiente Netz-/CDN-Fehler automatisch neu versuchen
  # Fortschrittsbalken nur im Terminal (im Log sonst zehntausende Zeilen).
  if [[ -t 1 ]]; then progress=(--progress-bar); else progress=(--no-progress-meter); fi

  set +e
  curl -L -C - --retry 5 --retry-delay 3 --retry-connrefused \
       "${progress[@]}" -o "$out" "$url"
  rc=$?
  set -e
  # Exit 33 = Server unterstuetzt kein Resume, weil die Datei schon komplett ist.
  [[ $rc -eq 33 ]] && rc=0
  return $rc
}

echo "$FILES" | while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  exp="${SIZES[$f]:-0}"

  fetch_one "$f"; rc=$?

  # Groesse gegenpruefen. Ein "erfolgreicher" curl-Lauf sagt NICHTS darueber,
  # ob die Datei stimmt (siehe Resume-Falle oben) — deshalb hart verifizieren.
  if [[ $rc -eq 0 && "$exp" -gt 0 ]]; then
    have=$(stat -f%z "${DEST}/${f}" 2>/dev/null || echo 0)
    if [[ "$have" -ne "$exp" ]]; then
      echo "   Groesse falsch ($have statt $exp) — einmaliger Neuversuch von vorn"
      rm -f "${DEST}/${f}"
      fetch_one "$f"; rc=$?
      have=$(stat -f%z "${DEST}/${f}" 2>/dev/null || echo 0)
      [[ $rc -eq 0 && "$have" -ne "$exp" ]] && rc=1
    fi
  fi

  if [[ $rc -ne 0 ]]; then
    echo "ERROR: Download fehlgeschlagen ($f, exit $rc). Skript erneut starten = Resume."
    exit $rc
  fi
done

echo
echo "Fertig. Groesse: $(du -sh "$DEST" | cut -f1)"
echo "Inhalt:"
ls -la "$DEST"
