#!/usr/bin/env bash
# Pull every repo in the workspace, then refresh the RAG index once.
#
# Built for the normal working pattern: pull everything before you start, work
# for a while, pull again tomorrow. Run it by hand, from a shell alias, or from
# an agent's session-start hook -- it is the same command either way.
#
# Handles SEVERAL repos, which is the whole point. A setup of five sibling
# checkouts under one directory has five places to pull from but ONE index; a
# single-repo script silently refreshes one of them and leaves you searching
# four stale trees with no indication anything is wrong.
#
# Safe enough to run unattended:
#   * never touches a dirty working tree -- uncommitted work is untouchable
#   * fast-forward only: never a merge commit, never a rewrite
#   * skips a detached HEAD or a branch with no upstream
#   * bounded network wait per repo; every failure is non-fatal and reported
#   * indexes incrementally, so only what actually changed is re-embedded
#
# Usage:  sync_and_index.sh [--full] [--no-pull] [--no-index] [--quiet]
#
# Env:
#   RAG_REPOS          ':'-separated repo list (default: auto-discovered)
#   RAG_SKIP_PULL=1    same as --no-pull
#   RAG_FETCH_TIMEOUT  seconds allowed for each fetch (default 20)
set -u

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$WORKSPACE/.poc-venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3 || true)"
[ -n "$PYTHON" ] || { echo "[RAG] no python3 available; skipping"; exit 0; }

FULL=""; DO_PULL=1; DO_INDEX=1; QUIET=0
for arg in "$@"; do
  case "$arg" in
    --full)     FULL="--full" ;;
    --no-pull)  DO_PULL=0 ;;
    --no-index) DO_INDEX=0 ;;
    --quiet|-q) QUIET=1 ;;
    -h|--help)  sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "[RAG] unknown option: $arg" >&2; exit 2 ;;
  esac
done
[ "${RAG_SKIP_PULL:-0}" = "1" ] && DO_PULL=0

REPO_ROOT="$("$PYTHON" "$WORKSPACE/rag_config.py" --repo-root 2>/dev/null || dirname "$WORKSPACE")"
FETCH_TIMEOUT="${RAG_FETCH_TIMEOUT:-20}"

say() { [ "$QUIET" = "1" ] || echo "$@"; }

# One repo. Echoes a single status word plus detail; never exits non-zero.
sync_one() {
  local repo="$1" name branch upstream behind before after changed
  name="$(basename "$repo")"

  if [ -n "$(git -C "$repo" status --porcelain 2>/dev/null)" ]; then
    printf '  %-22s %-12s %s\n' "$name" "skipped" "uncommitted changes — pull it yourself when ready"
    return
  fi
  branch="$(git -C "$repo" symbolic-ref --short -q HEAD || true)"
  if [ -z "$branch" ]; then
    printf '  %-22s %-12s %s\n' "$name" "skipped" "detached HEAD"; return
  fi
  upstream="$(git -C "$repo" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [ -z "$upstream" ]; then
    printf '  %-22s %-12s %s\n' "$name" "skipped" "'$branch' has no upstream"; return
  fi

  # Portable bounded fetch: background it and kill it if it overruns.
  git -C "$repo" fetch --quiet --prune 2>/dev/null &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$FETCH_TIMEOUT" ]; then
      kill "$pid" 2>/dev/null
      printf '  %-22s %-12s %s\n' "$name" "offline" "fetch exceeded ${FETCH_TIMEOUT}s"; return
    fi
    sleep 1; waited=$((waited + 1))
  done
  if ! wait "$pid" 2>/dev/null; then
    printf '  %-22s %-12s %s\n' "$name" "offline" "fetch failed; kept local state"; return
  fi

  behind="$(git -C "$repo" rev-list --count "HEAD..$upstream" 2>/dev/null || echo 0)"
  if [ "$behind" = "0" ]; then
    printf '  %-22s %-12s %s\n' "$name" "up to date" "$branch"; return
  fi

  before="$(git -C "$repo" rev-parse HEAD 2>/dev/null)"
  if git -C "$repo" merge --ff-only "$upstream" --quiet 2>/dev/null; then
    after="$(git -C "$repo" rev-parse HEAD 2>/dev/null)"
    changed="$(git -C "$repo" diff --name-only "$before..$after" 2>/dev/null | wc -l | tr -d ' ')"
    printf '  %-22s %-12s %s\n' "$name" "PULLED" \
      "$behind commit(s), $changed file(s) changed on $branch"
    echo "$repo" >> "$PULLED_LIST"
  else
    printf '  %-22s %-12s %s\n' "$name" "DIVERGED" \
      "cannot fast-forward $branch from $upstream — resolve manually"
  fi
}

PULLED_LIST="$(mktemp)"; trap 'rm -f "$PULLED_LIST"' EXIT

REPOS="$("$PYTHON" "$WORKSPACE/rag_config.py" --repos 2>/dev/null)"
NREPOS=0; [ -n "$REPOS" ] && NREPOS="$(printf '%s\n' "$REPOS" | wc -l | tr -d ' ')"

if [ "$DO_PULL" = "1" ]; then
  if [ "$NREPOS" = "0" ]; then
    say "[RAG] no git repositories found under $REPO_ROOT — indexing the tree as-is"
  else
    say "[RAG] syncing $NREPOS repo(s) under $REPO_ROOT"
    while IFS= read -r repo; do [ -n "$repo" ] && sync_one "$repo"; done <<< "$REPOS"
  fi
else
  say "[RAG] pull skipped"
fi

NPULLED=0; [ -s "$PULLED_LIST" ] && NPULLED="$(wc -l < "$PULLED_LIST" | tr -d ' ')"

if [ "$DO_INDEX" = "0" ]; then
  say "[RAG] indexing skipped (--no-index)"
  exit 0
fi

# Always index, even when nothing was pulled: your own uncommitted edits also
# make the index stale, and an incremental run over an unchanged tree is a
# sub-second no-op. The ingest itself is the cheapest honest staleness check.
if [ "$NPULLED" != "0" ]; then
  say "[RAG] $NPULLED repo(s) changed — re-indexing"
else
  say "[RAG] no remote changes — verifying the index is current"
fi
"$PYTHON" "$WORKSPACE/ingest_codebase.py" --target "$REPO_ROOT" $FULL 2>&1 | tail -3
