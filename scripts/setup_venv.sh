#!/usr/bin/env bash
set -euo pipefail

# Create a local virtualenv and install requirements
VENV_DIR=".venv"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install -r "$(dirname "$0")/../requirements.txt"

echo "Virtualenv created at $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
