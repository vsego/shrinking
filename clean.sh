#!/usr/bin/env bash

set -euo pipefail

rm -rf \
    .coverage \
    .pytest_cache \
    .ruff_cache \
    build \
    dist \
    htmlcov \
    .mypy_cache \
    src/*.egg-info \
    src/shrinking.egg-info
