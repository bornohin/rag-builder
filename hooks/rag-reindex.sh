#!/usr/bin/env bash
# Shared body for the git hooks that keep the RAG index in step with the tree.
# Incremental: only files whose SHA-1 changed are re-chunked and re-embedded.
#
# Targets the WORKSPACE root, not the repo this hook fired in. That distinction
# is the whole ballgame in a multi-repo setup: `git rev-parse --show-toplevel`
# from inside a hook returns the single repo being committed to, and indexing
# that alone silently DELETES every other repo's chunks from the shared index.
# rag_config owns the definition of the root, so ask it.
set -u
RAG_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$RAG_WORKSPACE/.poc-venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3 || true)"
[ -n "$PYTHON" ] || exit 0

REPO_ROOT="$("$PYTHON" "$RAG_WORKSPACE/rag_config.py" --repo-root 2>/dev/null || true)"
[ -n "$REPO_ROOT" ] || REPO_ROOT="$(cd "$RAG_WORKSPACE/.." && pwd)"

echo "[RAG] incremental re-index of $REPO_ROOT"
"$PYTHON" "$RAG_WORKSPACE/ingest_codebase.py" --target "$REPO_ROOT" 2>&1 | tail -3
exit 0
