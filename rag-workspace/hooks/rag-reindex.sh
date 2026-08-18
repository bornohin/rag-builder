#!/usr/bin/env bash
# Shared body for the git hooks that keep the RAG index in step with the tree.
# Incremental: only files whose SHA-1 changed are re-chunked and re-embedded.
set -u
RAG_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$RAG_WORKSPACE/.poc-venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3 || true)"
[ -n "$PYTHON" ] || exit 0

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$RAG_WORKSPACE/..")"
echo "[RAG] incremental re-index of $REPO_ROOT"
"$PYTHON" "$RAG_WORKSPACE/ingest_codebase.py" --target "$REPO_ROOT" 2>&1 | tail -4
exit 0
