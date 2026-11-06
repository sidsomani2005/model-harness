#!/usr/bin/env bash
# Start the WingMan demo UI with the project's own Python environment.
# Creates .venv and installs the project the first time. Extra args go to the server (e.g. --port 8800).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv and installing WingMan (first run only)..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -e ".[dev]"
fi
if ! .venv/bin/python -c "import anthropic, rank_bm25, medproof" 2>/dev/null; then
  echo "Installing missing dependencies into .venv..."
  .venv/bin/pip install -q -e ".[dev]"
fi
exec .venv/bin/python -m medproof.cli wingman "$@"
