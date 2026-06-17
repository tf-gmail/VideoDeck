#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Pick a Python executable.
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python not found. Install Python 3.11+ and retry."
  exit 1
fi

# Create a virtual environment if needed.
if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# Activate virtual environment for Linux/WSL or Git Bash fallback.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  echo "Could not find virtual environment activation script."
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

exec python main.py
