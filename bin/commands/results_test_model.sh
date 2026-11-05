#!/usr/bin/env bash
# Bash 4+: dispatch independent checkpoint evaluation using the active Python environment.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$script_dir/../helpers/test_saved_model.py" "$@"
