#!/usr/bin/env bash
# Install git hooks that keep the RAG index fresh after pull / commit /
# branch switch.
#
# Safe to re-run, and safe to run in a repo that already has hooks. Husky,
# pre-commit and lefthook all manage .git/hooks (or redirect it via
# core.hooksPath) and silently overwriting their files breaks a developer's
# commit pipeline in a way that is very hard to trace back to a RAG installer.
# So: honour core.hooksPath, and where a foreign hook already exists, CHAIN it
# rather than replace it -- the existing hook keeps running, ours runs after,
# and its exit status can never fail the git operation.
set -euo pipefail

RAG_WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
MARKER="rag-reindex.sh"

PYTHON="$RAG_WORKSPACE/.poc-venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3 || true)"
REPO_ROOT="$("$PYTHON" "$RAG_WORKSPACE/rag_config.py" --repo-root 2>/dev/null || dirname "$RAG_WORKSPACE")"

# A workspace is often several sibling checkouts, not one tree. Installing into
# only the first would leave the others pulling with no re-index and no warning
# -- stale results that look perfectly healthy. So install into every repo the
# same discovery logic the indexer and the sync script use.
REPOS="$("$PYTHON" "$RAG_WORKSPACE/rag_config.py" --repos 2>/dev/null || true)"
[ $# -gt 0 ] && REPOS="$(printf '%s\n' "$@")"
if [ -z "${REPOS//[[:space:]]/}" ]; then
  echo "No git repository found under $REPO_ROOT." >&2
  echo "Pass repo paths explicitly, or set RAG_REPOS." >&2
  exit 1
fi

install_into() {
  local REPO_ROOT="$1"

# core.hooksPath wins over .git/hooks whenever it is set (this is exactly what
# husky does), so writing to .git/hooks there would install files git ignores.
HOOKS_PATH="$(git -C "$REPO_ROOT" config --get core.hooksPath 2>/dev/null || true)"
if [ -n "${RAG_GIT_DIR:-}" ]; then
  HOOK_DIR="$RAG_GIT_DIR/hooks"
elif [ -n "$HOOKS_PATH" ]; then
  case "$HOOKS_PATH" in
    /*) HOOK_DIR="$HOOKS_PATH" ;;
    *)  HOOK_DIR="$REPO_ROOT/$HOOKS_PATH" ;;
  esac
  echo "Note: core.hooksPath is set -> installing into $HOOK_DIR"
else
  GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --git-dir 2>/dev/null || echo "$REPO_ROOT/.git")"
  case "$GIT_DIR" in
    /*) HOOK_DIR="$GIT_DIR/hooks" ;;
    *)  HOOK_DIR="$REPO_ROOT/$GIT_DIR/hooks" ;;
  esac
fi

if ! git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  echo "  $(basename "$REPO_ROOT"): not a git repository — skipped" >&2
  return 0
fi
mkdir -p "$HOOK_DIR"
echo "  $(basename "$REPO_ROOT"):"

installed=""
for hook in post-merge post-commit post-checkout post-rewrite; do
  target="$HOOK_DIR/$hook"
  chained=""

  # Snapshot a foreign hook exactly once. On every later run the marker is
  # present, so this branch is skipped -- which is why the chain is rebuilt
  # from the saved file below rather than from this check. Getting that wrong
  # silently drops the user's original hook on the SECOND install, not the first.
  if [ -e "$target" ] && ! grep -q "$MARKER" "$target" 2>/dev/null; then
    if [ ! -e "$target.pre-rag" ]; then
      cp "$target" "$target.pre-rag"
      chmod +x "$target.pre-rag" 2>/dev/null || true
    fi
    echo "    chained existing $hook -> $hook.pre-rag (it still runs first)"
  fi
  # Re-assert the chain on every run, whoever wrote the current hook file.
  [ -e "$target.pre-rag" ] && chained="$target.pre-rag"

  {
    echo '#!/usr/bin/env bash'
    echo "# Installed by rag-workspace/install_hooks.sh — keeps the RAG index fresh."
    if [ -n "$chained" ]; then
      echo "# A pre-existing $hook was found and is chained below, unmodified."
      echo "if [ -x \"$chained\" ]; then \"$chained\" \"\$@\" || exit \$?; fi"
    fi
    # Never let indexing fail the git operation the user actually asked for.
    echo "\"$RAG_WORKSPACE/hooks/rag-reindex.sh\" || true"
    echo "exit 0"
  } > "$target"
  chmod +x "$target"
  installed="$installed $hook"
done
echo "    installed:$installed"
}

echo "Installing RAG hooks into:"
while IFS= read -r repo; do
  [ -n "${repo//[[:space:]]/}" ] && install_into "$repo"
done <<< "$REPOS"

echo
echo "Done. Every repo above re-indexes the shared index after pull, commit,"
echo "checkout and rebase. For the manual path, use: $RAG_WORKSPACE/sync_and_index.sh"
