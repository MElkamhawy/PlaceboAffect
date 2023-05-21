#!/bin/sh
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate PlaceboAffect
fi
python3 "$REPO_DIR/src/main.py" "$@"