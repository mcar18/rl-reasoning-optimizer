#!/usr/bin/env bash
# Setup script for rl-reasoning-optimizer
# Run from project root: bash setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating virtual environment..."
python -m venv .venv

echo "Activating virtual environment..."
if [ -f .venv/Scripts/activate ]; then
  # Windows (Git Bash / MSYS)
  source .venv/Scripts/activate
elif [ -f .venv/bin/activate ]; then
  # Unix
  source .venv/bin/activate
else
  echo "Could not find activate script."
  exit 1
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Done. Activate the venv with:"
echo "  PowerShell:  .venv\\Scripts\\Activate.ps1"
echo "  CMD:        .venv\\Scripts\\activate.bat"
echo "  Git Bash:   source .venv/Scripts/activate"
echo "  Unix:       source .venv/bin/activate"
