#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-python3}"
mode="${1:-online}"
if [[ "$mode" != online && "$mode" != --offline ]]; then
  echo "Usage: bash scripts/setup_env.sh [--offline]" >&2
  exit 2
fi
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 12), "Use Python 3.12+ for the pinned requirements"'

if [[ "$mode" == --offline ]]; then
  # Standard venv fallback: reuse packages already installed in the base Python.
  "$python_bin" -m venv --system-site-packages .venv
  .venv/bin/python -m pip install --no-cache-dir --no-index --no-build-isolation --no-deps -e .
else
  # Re-running venv without --system-site-packages disables inherited packages.
  "$python_bin" -m venv .venv
  .venv/bin/python -m pip install --no-cache-dir --upgrade pip
  .venv/bin/python -m pip install --no-cache-dir -r requirements.txt
  .venv/bin/python -m pip check
fi

.venv/bin/python scripts/check_env.py
echo "Activate with: source .venv/bin/activate"
