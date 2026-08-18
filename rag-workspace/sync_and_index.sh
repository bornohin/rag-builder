#!/usr/bin/env bash
# Refresh the working tree and the RAG index at the start of an agent session.
#
# Client-agnostic: call it from a Claude Code SessionStart hook, a Gemini/Cursor
# equivalent, a shell alias, or by hand. Designed to be safe enough to run
# unattended and fast enough to block a session start on:
#
#   * never touches a dirty working tree (your uncommitted work is untouchable)
#   * fast-forward only -- never creates a merge commit, never rewrites history
#   * skips the pull on a detached HEAD or a branch with no upstream
#   * short network timeout, and every failure is non-fatal
#   * re-indexes incrementally (~1s when nothing changed)
#
# Env:
#   RAG_SKIP_PULL=1     refresh the index but do not touch git
#   RAG_FETCH_TIMEOUT   seconds to allow for the network fetch (default 20)
set -u

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$WORKSPACE/.poc-venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3 || true)"
[ -n "$PYTHON" ] || { echo "[RAG] no python3 available; skipping"; exit 0; }

REPO_ROOT="$(git -C "$WORKSPACE" rev-parse --show-toplevel 2>/dev/null || dirname "$WORKSPACE")"
FETCH_TIMEOUT="${RAG_FETCH_TIMEOUT:-20}"

pull_latest() {
  [ "${RAG_SKIP_PULL:-0}" = "1" ] && { echo "[RAG] pull skipped (RAG_SKIP_PULL=1)"; return; }
  git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 || { echo "[RAG] not a git repo; skipping pull"; return; }

  local branch upstream
  branch="$(git -C "$REPO_ROOT" symbolic-ref --short -q HEAD || true)"
  if [ -z "$branch" ]; then
    echo "[RAG] detached HEAD; skipping pull"; return
  fi
  upstream="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [ -z "$upstream" ]; then
    echo "[RAG] '$branch' has no upstream; skipping pull"; return
  fi
  if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
    echo "[RAG] working tree has uncommitted changes; skipping pull (run 'git pull' yourself when ready)"
    return
  fi

  # Portable timeout: background the fetch and kill it if it overruns.
  git -C "$REPO_ROOT" fetch --quiet --prune 2>/dev/null &
  local fetch_pid=$! waited=0
  while kill -0 "$fetch_pid" 2>/dev/null; do
    [ "$waited" -ge "$FETCH_TIMEOUT" ] && { kill "$fetch_pid" 2>/dev/null; echo "[RAG] fetch timed out after ${FETCH_TIMEOUT}s"; return; }
    sleep 1; waited=$((waited + 1))
  done
  wait "$fetch_pid" 2>/dev/null || { echo "[RAG] fetch failed (offline?); continuing"; return; }

  local behind
  behind="$(git -C "$REPO_ROOT" rev-list --count "HEAD..$upstream" 2>/dev/null || echo 0)"
  if [ "$behind" = "0" ]; then
    echo "[RAG] already up to date with $upstream"
    return
  fi
  if git -C "$REPO_ROOT" merge --ff-only "$upstream" --quiet 2>/dev/null; then
    echo "[RAG] fast-forwarded $branch by $behind commit(s) from $upstream"
  else
    echo "[RAG] cannot fast-forward $branch (diverged from $upstream) -- resolve manually"
  fi
}

pull_latest
# The git hooks also fire on a successful merge; this run is idempotent and
# becomes a ~1s no-op when the hook already did the work.
"$PYTHON" "$WORKSPACE/ingest_codebase.py" --target "$REPO_ROOT" 2>&1 | tail -3
