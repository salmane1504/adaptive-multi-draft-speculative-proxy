#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Setting up Speculative Proxy Environment ==="

# 1. Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install it first."
    exit 1
fi

# 2. Create the virtual environment if it doesn't exist
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment already exists."
fi

# 3. Activate the virtual environment
# Note: When running a script, 'source' only affects the script's subshell.
# We print instructions at the end for the user.
source $VENV_DIR/bin/activate

# 4. Install dependencies
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
# Using a dummy install command here; in reality, it reads your requirements.txt
pip install -r requirements.txt

echo "=== Setup Complete ==="
echo "To activate the environment in your terminal, run:"
echo "source $VENV_DIR/bin/activate"