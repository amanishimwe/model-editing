#!/usr/bin/env bash
# Run E1 through E37. Extra arguments are passed through, for example --no-live.
cd "$(dirname "$0")/.."
exec .venv/bin/python -m selfedit run all "$@"
