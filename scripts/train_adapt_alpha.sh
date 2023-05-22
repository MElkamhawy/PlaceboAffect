#!/bin/sh
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
"$SCRIPT_DIR/model_runner.sh" -m train -t adaptation -s alpha