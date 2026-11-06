#!/usr/bin/env bash
# Clean-install smoke test of the submission zip: unzip into a fresh temp dir,
# create a new virtualenv, install, run tests, validate cases and build the report.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="${1:-$ROOT/submission/medproof_submission.zip}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unzip -q "$ZIP" -d "$WORK"
cd "$WORK/medproof"
test ! -e .env || { echo "FAIL: .env is inside the zip"; exit 1; }
python3 -m venv .venv
.venv/bin/pip install -q ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m medproof.cli validate-cases > /dev/null
.venv/bin/python -m medproof.cli report --output report/index.html
echo "SMOKE TEST PASSED in clean install at $WORK/medproof"
