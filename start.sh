#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${VIDEODECK_HOST:-127.0.0.1}"
PORT="${VIDEODECK_PORT:-8000}"

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

# Stop an already running server process on the target port.
if [[ "${OS:-}" == "Windows_NT" ]]; then
  powershell -NoProfile -Command "\
    \$p=(Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess); \
    if (\$p) { Write-Output \"Stopping existing server PID \$p on port $PORT\"; Stop-Process -Id \$p -Force }\
  " >/dev/null 2>&1 || true
else
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      echo "Stopping existing server on port $PORT ($pids)"
      kill -9 $pids || true
    fi
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "$PORT"/tcp >/dev/null 2>&1 || true
  fi
fi

exec python main.py
