#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PACKAGE_IMPORT_NAME="shrinking"
PYTHONWARNINGS="ignore::PendingDeprecationWarning${PYTHONWARNINGS:+:$PYTHONWARNINGS}"

if [ "${USE_INSTALLED_PACKAGE:-0}" = "1" ]; then
    COVERAGE_SOURCE="$PACKAGE_IMPORT_NAME"
    PYTHONPATH_VALUE="${PYTHONPATH:-}"
else
    COVERAGE_SOURCE="$SCRIPT_DIR/src"
    PYTHONPATH_VALUE="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
fi

if python -c "import coverage" >/dev/null 2>&1; then
    PYTHONWARNINGS="$PYTHONWARNINGS" \
    PYTHONPATH="$PYTHONPATH_VALUE" \
    python -m coverage run --source "$COVERAGE_SOURCE" -m unittest discover -s "$SCRIPT_DIR/tests" -t "$SCRIPT_DIR"
    python -m coverage report
    python -m coverage html
else
    echo "coverage is not installed; running tests without coverage" >&2
    PYTHONWARNINGS="$PYTHONWARNINGS" \
    PYTHONPATH="$PYTHONPATH_VALUE" \
    python -m unittest discover -s "$SCRIPT_DIR/tests" -t "$SCRIPT_DIR"
fi
