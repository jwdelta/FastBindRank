#!/usr/bin/env bash

set -euo pipefail

original_args=("$@")
lock_fd=""

usage() {
  cat <<'EOF'
Usage:
  fastbindrank library prepare-pubchem [--workspace PATH]
                                       [--python PATH]
                                       [--chunk-lines N]
                                       [--workers N]
                                       [--fingerprint-script PATH]
                                       [--skip-download]
                                       [--skip-split]
                                       [--skip-fingerprints]
                                       [--cleanup-downloads]

Description:
  Prepare a screening library from PubChem CID-SMILES by:
  1. downloading CID-SMILES.gz and CID-SMILES.gz.md5
  2. verifying the checksum when possible
  3. decompressing CID-SMILES.gz into CID-SMILES
  4. creating library/smiles.txt and splitting it into library/smiles_chunks/
  5. generating Morgan fingerprints into library/morgan_fingerprints/
  6. building/reusing library/molecule_index.sqlite for canonical deduplication and scaffold sampling

Parameters:
  --workspace PATH         Output workspace. Default: current directory
  --python PATH            Python executable to use. Default: python3
  --chunk-lines N          Lines per split chunk. Default: 1000000
  --workers N              Number of worker processes. Default: 8
  --fingerprint-script     Override the bundled Morgan fingerprint helper
  --skip-download          Reuse existing downloaded or decompressed PubChem files
  --skip-split             Reuse existing smiles.txt and smiles_chunks/
  --skip-fingerprints      Skip Morgan fingerprint generation
  --cleanup-downloads      Remove the pubchem download directory after library generation
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$(cd "$script_dir/.." && pwd)"
workspace="$(pwd)"
python_bin="python3"
chunk_lines="1000000"
workers="8"
fingerprint_script="$bin_dir/helpers/morgan_fp.py"
skip_download="false"
skip_split="false"
skip_fingerprints="false"
cleanup_downloads="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lock-fd) lock_fd="$2"; shift 2 ;;
    --workspace)
      workspace="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --chunk-lines)
      chunk_lines="$2"
      shift 2
      ;;
    --workers)
      workers="$2"
      shift 2
      ;;
    --fingerprint-script)
      fingerprint_script="$2"
      shift 2
      ;;
    --skip-download)
      skip_download="true"
      shift 1
      ;;
    --skip-split)
      skip_split="true"
      shift 1
      ;;
    --skip-fingerprints|--skip-morgan)
      skip_fingerprints="true"
      shift 1
      ;;
    --cleanup-downloads)
      cleanup_downloads="true"
      shift 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$skip_download" == "false" ]] && ! command -v wget >/dev/null 2>&1; then
  echo "Error: wget is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$fingerprint_script" ]]; then
  echo "Error: fingerprint helper not found: $fingerprint_script" >&2
  exit 1
fi

# Hold one lock through sampling/library generation and final verification.
if [[ -z "$lock_fd" ]]; then
  exec "$python_bin" "$bin_dir/helpers/locked_command.py" \
    --lock-file "$workspace/.library_prepare.lock" -- bash "$script_dir/library_prepare_pubchem.sh" "${original_args[@]}"
fi
"$python_bin" - "$bin_dir/helpers" "$lock_fd" "$workspace/.library_prepare.lock" <<'PYLOCK'
import sys
sys.path.insert(0, sys.argv[1])
from locked_command import check_lock
check_lock(int(sys.argv[2]), sys.argv[3])
PYLOCK

# An indexed library is immutable: reject rebuilding before touching any assets.
if [[ -f "$workspace/library/molecule_index.sqlite" ]] && \
   [[ "$skip_download" != "true" || "$skip_split" != "true" || "$skip_fingerprints" != "true" ]]; then
  echo "Error: indexed library already exists. To verify/reuse it, pass --skip-download --skip-split --skip-fingerprints. To rebuild, use a new workspace." >&2
  exit 1
fi

pubchem_gz_url="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
pubchem_md5_url="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz.md5"

download_dir="$workspace/pubchem"
library_dir="$workspace/library"
compressed_file="$download_dir/CID-SMILES.gz"
checksum_file="$download_dir/CID-SMILES.gz.md5"
decompressed_file="$download_dir/CID-SMILES"
smiles_table="$library_dir/smiles.txt"
smiles_chunks_dir="$library_dir/smiles_chunks"
morgan_dir="$library_dir/morgan_fingerprints"

mkdir -p "$workspace" "$download_dir" "$library_dir"

if [[ "$skip_download" == "false" ]]; then
  echo "Downloading CID-SMILES.gz ..."
  wget -O "$compressed_file" "$pubchem_gz_url"

  echo "Downloading CID-SMILES.gz.md5 ..."
  wget -O "$checksum_file" "$pubchem_md5_url"

  expected_md5="$(awk '{print $1}' "$checksum_file")"

  if command -v md5sum >/dev/null 2>&1; then
    actual_md5="$(md5sum "$compressed_file" | awk '{print $1}')"
  elif command -v md5 >/dev/null 2>&1; then
    actual_md5="$(md5 -q "$compressed_file")"
  else
    actual_md5=""
  fi

  if [[ -n "$actual_md5" ]]; then
    if [[ "$actual_md5" != "$expected_md5" ]]; then
      echo "Error: md5 mismatch for CID-SMILES.gz" >&2
      echo "Expected: $expected_md5" >&2
      echo "Actual:   $actual_md5" >&2
      exit 1
    fi
    echo "MD5 verification passed."
  else
    echo "Warning: no md5 tool found; skipping checksum verification."
  fi
else
  echo "Skipping download step."
fi

if [[ "$skip_split" == "false" ]] && { [[ "$skip_download" == "false" ]] || [[ ! -f "$decompressed_file" ]]; }; then
  if [[ ! -f "$compressed_file" ]]; then
    echo "Error: missing $compressed_file. Remove --skip-download or provide the file." >&2
    exit 1
  fi
  echo "Decompressing CID-SMILES.gz ..."
  gzip -dc "$compressed_file" > "$decompressed_file"
elif [[ "$skip_split" == "false" ]]; then
  echo "Reusing existing decompressed file: $decompressed_file"
fi

if [[ "$skip_split" == "false" ]]; then
  echo "Generating library/smiles.txt ..."
  awk '{print $2 "\t" $1}' "$decompressed_file" > "$smiles_table"

  echo "Splitting smiles.txt into library/smiles_chunks/ ..."
  rm -rf "$smiles_chunks_dir"
  mkdir -p "$smiles_chunks_dir"
  "$python_bin" - "$smiles_table" "$smiles_chunks_dir" "$chunk_lines" <<'PYCODE'
import sys
from pathlib import Path
source, directory, chunk_lines = sys.argv[1:]
chunk_lines = int(chunk_lines)
if chunk_lines < 1:
    raise ValueError('--chunk-lines must be positive')
output = None
try:
    with open(source) as handle:
        for n, line in enumerate(handle):
            if n % chunk_lines == 0:
                if output is not None:
                    output.close()
                output = (Path(directory) / f'smile_all_{n // chunk_lines:03d}.txt').open('w')
            output.write(line)
finally:
    if output is not None:
        output.close()
PYCODE
else
  echo "Skipping split step."
  if [[ ! -f "$smiles_table" || ! -d "$smiles_chunks_dir" ]]; then
    echo "Error: --skip-split was given, but smiles.txt or smiles_chunks/ is missing." >&2
    exit 1
  fi
fi

if [[ "$skip_fingerprints" == "false" ]]; then
  echo "Generating Morgan fingerprints into library/morgan_fingerprints/ ..."
  rm -rf "$morgan_dir"
  mkdir -p "$morgan_dir"
  "$python_bin" "$fingerprint_script" \
    --smile_folder_path "$smiles_chunks_dir" \
    --folder_name "$morgan_dir" \
    --tot_process "$workers"
else
  echo "Skipping Morgan fingerprint generation."
fi

"$python_bin" "$bin_dir/helpers/build_molecule_index.py" \
  --smiles-file "$smiles_table" --morgan-directory "$morgan_dir" \
  --output "$library_dir/molecule_index.sqlite" --workers "$workers"

if [[ "$cleanup_downloads" == "true" ]]; then
  echo "Removing PubChem download directory: $download_dir"
  rm -rf "$download_dir"
fi

echo "Done."
echo "Workspace:         $workspace"
if [[ "$cleanup_downloads" == "false" ]]; then
  echo "PubChem files:     $download_dir"
fi
echo "SMILES table:      $smiles_table"
echo "SMILES chunks:     $smiles_chunks_dir"
echo "Morgan library:    $morgan_dir"
echo "Molecule index:    $library_dir/molecule_index.sqlite"
