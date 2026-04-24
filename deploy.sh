#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CURRENT_BRANCH="$(git branch --show-current)"
RELEASE_BRANCH="${RELEASE_BRANCH:-}"
AHEAD=0

if [[ -n "$RELEASE_BRANCH" && "$CURRENT_BRANCH" != "$RELEASE_BRANCH" ]]; then
    echo "Refusing to deploy from branch '$CURRENT_BRANCH'." >&2
    echo "Expected branch: '$RELEASE_BRANCH'." >&2
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Refusing to deploy with uncommitted changes." >&2
    exit 1
fi

if git rev-parse --verify "@{upstream}" >/dev/null 2>&1; then
    read -r BEHIND AHEAD < <(git rev-list --left-right --count "@{upstream}...HEAD")
    if [[ "$BEHIND" != "0" ]]; then
        echo "Refusing to deploy: local branch is behind its upstream." >&2
        exit 1
    fi
fi

./run_tests.sh
ruff check .
pyright
python -m build --no-isolation
python -m twine check dist/*

if git rev-parse --verify "@{upstream}" >/dev/null 2>&1; then
    if [[ "$AHEAD" != "0" ]]; then
        git push
    fi
fi

hatch publish
