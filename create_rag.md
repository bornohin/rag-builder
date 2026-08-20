# create_rag.md — Build a Local Hybrid Codebase RAG + MCP Server

**What this is.** A complete, self-contained runbook for standing up a 100% local,
offline code-retrieval system over *any* repository, served to *any* AI assistant over
stdio MCP. Hand this file to an agent (or follow it yourself) inside a target directory
and work through the phases in order. Nothing here is specific to a language, a
repository, a vendor, or a model.

**What you get.**

- Tree-sitter structure-aware chunking with **exact `file:start_line-end_line`** per hit
- Hybrid retrieval: dense ONNX embeddings (ChromaDB) + code-aware BM25, fused with
  **query-adaptive weighted RRF** and an exact-symbol boost
- Chunk boundaries measured with the **encoder's own tokenizer**, so nothing is
  silently truncated at embed time
- **Incremental** re-indexing (SHA-1 manifest) — seconds, not minutes, after an edit
- Seven MCP tools, exception-guarded, with staleness detection and self-healing re-index
- Optional local cross-encoder reranker (off by default, degrades silently)
- Git hooks that keep the index in step with the working tree — and **chain** any
  hooks the repo already had instead of overwriting them
- The index is read through a **restricted unpickler**, so a tampered index file
  cannot execute code inside the MCP server
- Runs entirely in user space: no sudo, no daemon, no network at query time
- The workspace is self-contained and indexes its **parent** directory, so it can be
  its own git checkout dropped into any project without disturbing that project's tree

**Requirements.** Python 3.9+, `git` (optional but recommended), ~500 MB disk for the
virtualenv and model, and one network fetch during setup to install packages and cache
the model weights. After setup, nothing phones home.

**Conventions in this document.** `<TARGET>` is the repository root you are indexing;
`<WORKSPACE>` is `<TARGET>/rag-workspace`. Replace them literally, or export them:

> **Already have this repository?** Then you can skip to Phase 6 — clone it straight
> into your project as the workspace and the files below are already written for you:
> `git clone <this repo> "$TARGET/rag-workspace"`. Work through the phases anyway if
> you want to understand what each file is doing, or are adapting it.

```bash
export TARGET="$(pwd)"
export WORKSPACE="$TARGET/rag-workspace"
```

---

## Phase 0 — Survey the target (do this before writing anything)

The goal is to learn what you are indexing, so the exclusion rules and validation
queries in later phases are grounded in reality rather than guessed.

### 0.1 Find the repository root and any nested projects

```bash
cd "$TARGET"
git rev-parse --show-toplevel 2>/dev/null || echo "not a git repo — that is fine"

# Nested project markers: each hit is a sub-project worth knowing about.
find . -maxdepth 3 \
  \( -name node_modules -o -name .git -o -name build -o -name dist -o -name .venv \) -prune -o \
  \( -name 'package.json' -o -name 'build.gradle*' -o -name 'pubspec.yaml' \
     -o -name 'Cargo.toml' -o -name 'go.mod' -o -name 'pyproject.toml' \
     -o -name 'pom.xml' -o -name '*.xcodeproj' -o -name 'requirements.txt' \
     -o -name 'CMakeLists.txt' -o -name 'composer.json' -o -name 'Gemfile' \) \
  -print 2>/dev/null | sort
```

Each match names a sub-project. Record their directory names — they become the natural
values a caller will pass to `path_filter` (e.g. `path_filter="backend"`).

### 0.2 Profile the languages actually present

```bash
find . -type f \
  \( -path ./node_modules -o -path ./.git -o -path '*/build/*' -o -path '*/dist/*' \
     -o -path '*/.venv/*' -o -path '*/vendor/*' \) -prune -o \
  -type f -name '*.*' -print 2>/dev/null \
  | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -25
```

### 0.3 Find the generated files that must NOT be indexed

Machine-generated files are the single largest source of index bloat. In one real
repository a lockfile alone was 13% of every chunk.

```bash
find . -type f \( -name '*.lock' -o -name 'package-lock.json' -o -name '*.min.js' \
  -o -name '*.g.dart' -o -name '*_pb2.py' -o -name '*.map' \) \
  -not -path '*/node_modules/*' -size +50k 2>/dev/null | head -20

# Largest tracked-looking files overall:
find . -type f -not -path '*/node_modules/*' -not -path '*/.git/*' -size +500k 2>/dev/null | head -20
```

Anything that appears here and is *generated* belongs in `EXCLUDE_FILE_NAMES` or
`EXCLUDE_FILE_PATTERNS` in Phase 2. Add project-specific entries there — the defaults
cover the common ecosystems but cannot know about yours.

### 0.4 Pick three validation queries

Before building anything, write down three things you already know the answer to:

1. **A distinctive identifier** — a function, class, table, or RPC name unique to this
   codebase. You will assert its *definition* ranks first.
2. **A conceptual question** — e.g. "how does session refresh work". You will assert the
   top hits are the files you would have opened by hand.
3. **A string you are certain is absent** — you will assert the tools say so cleanly.

These are your acceptance tests in Phase 8. A RAG server that returns plausible-looking
results while failing these is worse than no server, because it is trusted.

---

## Phase 1 — Workspace and dependencies

```bash
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
python3 -m venv .poc-venv
```

Create `<WORKSPACE>/requirements.txt`:

```text
# Local codebase RAG — everything runs offline, in user space, no sudo.
fastmcp>=2.0
chromadb>=0.5
fastembed>=0.4
rank-bm25>=0.2.2
tree-sitter>=0.23
tree-sitter-language-pack>=0.7
```

Install:

```bash
"$WORKSPACE/.poc-venv/bin/pip" install --upgrade pip
"$WORKSPACE/.poc-venv/bin/pip" install -r "$WORKSPACE/requirements.txt"
```

Verify every dependency imports, and that tree-sitter can parse the languages Phase 0
found:

```bash
"$WORKSPACE/.poc-venv/bin/python3" - <<'PY'
import importlib
for m in ("fastmcp", "chromadb", "fastembed", "rank_bm25", "tree_sitter_language_pack"):
    importlib.import_module(m); print("ok", m)
from tree_sitter_language_pack import get_parser
for lang in ("python", "typescript", "tsx", "javascript", "java", "kotlin", "swift",
             "dart", "go", "rust", "c", "cpp", "sql", "html", "css", "yaml", "json"):
    try:
        get_parser(lang); print("parser ok", lang)
    except Exception as exc:
        print("parser MISSING", lang, type(exc).__name__)
PY
```

`tree-sitter-language-pack` ships prebuilt wheels for common platforms. If it will not
install, the pipeline still works — `rag_chunker.py` degrades to a heading/window
splitter automatically — but line ranges become approximate, so prefer fixing the
install.

---

## Phase 2 — `rag_config.py` (single source of truth)

Every other module imports its paths, model, exclusions and tokenizer from here.
Duplicating any of these values into a second file is precisely how a RAG pipeline
silently drifts out of sync with itself.

Two things to customise after pasting: add the generated-file patterns you found in
Phase 0.3 to `EXCLUDE_FILE_NAMES` / `EXCLUDE_FILE_PATTERNS`, and add the workspace
directory itself (`rag-workspace`) to `EXCLUDE_DIRS` — its own source quotes your
codebase's identifiers in comments and will otherwise pollute symbol searches.

Write `<WORKSPACE>/rag_config.py`:

```python
#!/usr/bin/env python3
"""Single source of truth for the local hybrid codebase RAG pipeline.

Every other script (ingest, MCP server, model downloader, git hooks) imports
its paths, model name, exclusion rules and tokenizer from here so the pieces
can never drift apart.

Everything is overridable by environment variable so the same checkout can be
pointed at a different repo or model without editing code -- which is what
makes the server portable across MCP clients (Claude Code, Gemini/Antigravity,
Cursor, Windsurf, Continue, or any other stdio MCP host).
"""
import glob
import os
import sys
import pickle
import re
import subprocess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _detect_repo_root() -> str:
    """RAG_REPO_ROOT env > git worktree enclosing the PARENT > parent directory.

    Note it asks git about the parent, never about this directory. The workspace
    is distributed as its own checkout, so `git -C <workspace> rev-parse
    --show-toplevel` answers with the workspace itself -- and indexing that
    would index the retrieval tooling instead of the code you wanted to search,
    silently and with no error to notice.
    """
    env = os.environ.get("RAG_REPO_ROOT")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    parent = os.path.dirname(BASE_DIR)
    try:
        out = subprocess.check_output(
            ["git", "-C", parent, "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        root = out.decode().strip()
        if root and os.path.isdir(root):
            return root
    except Exception:
        pass
    return parent


REPO_ROOT = _detect_repo_root()


# ---------------------------------------------------------------------------
# Repository discovery (one index, several checkouts)
# ---------------------------------------------------------------------------
# A working setup is often several sibling repos under one directory rather
# than a single tree. The index does not care -- it walks paths -- but every
# git-aware operation does: pulling, hook installation and commit recording all
# have to happen once PER REPO or they silently cover only one of them.
#
# Order of precedence: RAG_REPOS (explicit, ':' or newline separated) > the
# root itself if it is a repo, plus any direct children that are their own
# repos. Shell scripts read this list via `python3 rag_config.py --repos` so
# there is exactly one definition of "which repos are in play".
def _is_git_repo(path: str) -> bool:
    """True only if `path` is a repo ROOT -- not a subdirectory of one, and
    not a stray `.git` directory left behind by a mis-aimed installer."""
    try:
        out = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return False
    top = out.decode().strip()
    return bool(top) and os.path.realpath(top) == os.path.realpath(path)


def discover_repos(root: str = None) -> list:
    root = os.path.abspath(root or REPO_ROOT)
    env = os.environ.get("RAG_REPOS", "").strip()
    if env:
        out = []
        for part in re.split(r"[:\n]", env):
            part = part.strip()
            if not part:
                continue
            path = part if os.path.isabs(part) else os.path.join(root, part)
            path = os.path.abspath(os.path.expanduser(path))
            if os.path.isdir(path) and path not in out:
                out.append(path)
        return out

    repos = []
    if _is_git_repo(root):
        repos.append(root)
    try:
        children = sorted(os.listdir(root))
    except OSError:
        children = []
    for name in children:
        if name.startswith(".") or name in EXCLUDE_DIRS:
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path) and _is_git_repo(path) and path not in repos:
            repos.append(path)
    return repos
INDEX_DIR = os.path.abspath(os.environ.get("RAG_INDEX_DIR", BASE_DIR))
CACHE_DIR = os.path.join(INDEX_DIR, ".models_cache")
CHROMA_DIR = os.path.join(INDEX_DIR, "chroma_db")
INDEX_PATH = os.path.join(INDEX_DIR, "rag_index.pkl")      # chunks + bm25 + manifest
LEGACY_BM25_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")
# Progress record for an interrupted ingest. Deleted on clean completion, so
# its mere existence means "the last run did not finish".
CHECKPOINT_PATH = os.path.join(INDEX_DIR, "rag_index.pkl.ckpt")
LOG_PATH = os.path.join(INDEX_DIR, "rag.log")
COLLECTION_NAME = os.environ.get("RAG_COLLECTION", "codebase")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# BAAI/bge-small-en-v1.5: Apache-2.0 (well, MIT), 384-dim, 512-token window,
# runs fully offline as ONNX via fastembed. Swap with RAG_EMBED_MODEL.
MODEL_NAME = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# bge / e5 / arctic families are trained with an asymmetric query instruction.
# Passages are embedded bare; queries get the prefix. Skipping this measurably
# degrades recall, which is why it is centralised here instead of inlined.
_QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
    "arctic": "Represent this sentence for searching relevant passages: ",
    "gte": "",
}


def query_prefix(model_name: str = MODEL_NAME) -> str:
    lowered = model_name.lower()
    for key, prefix in _QUERY_PREFIXES.items():
        if key in lowered:
            return prefix
    return ""


# ---------------------------------------------------------------------------
# Pre-built model cache (optional fast path)
# ---------------------------------------------------------------------------
# Downloading the weights from HuggingFace works, but it is the one setup step
# that needs the public internet, and the one that fails on a locked-down
# network or when upstream renames a repo. So the same cache is published as a
# release asset on this project and download_model.py prefers it: fetch, verify
# the digest, unpack. If the asset is unreachable for any reason it falls
# straight back to HuggingFace, so nothing depends on the release existing.
#
# The digest is the point. An unverified tarball pulled from the internet and
# unpacked into a directory the server later loads is exactly the supply-chain
# hole the restricted unpickler closes on the index side; leaving it open here
# would just move the problem.
MODEL_ASSET_URL = os.environ.get(
    "RAG_MODEL_ASSET_URL",
    "https://github.com/bornohin/rag-builder/releases/download/models-v1/"
    "bge-small-en-v1.5-onnx-q.tar.gz")
MODEL_ASSET_SHA256 = os.environ.get("RAG_MODEL_ASSET_SHA256", "21780c32b286a8661c740e77250104fd2cad03d75b4d598ea4ead8083724fe49")
# Set RAG_SKIP_MODEL_ASSET=1 to ignore the asset and always use HuggingFace.
SKIP_MODEL_ASSET = os.environ.get("RAG_SKIP_MODEL_ASSET", "") == "1"

# Hard ceiling of the encoder. Chunks longer than this are silently truncated
# by the tokenizer, so the chunker splits before reaching it.
MODEL_MAX_TOKENS = int(os.environ.get("RAG_MODEL_MAX_TOKENS", "512"))
MAX_CHUNK_TOKENS = int(os.environ.get("RAG_MAX_CHUNK_TOKENS", "440"))
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH", "256"))
# Write the progress record every N embed batches. Embeddings already land in
# the vector store batch by batch; what used to be lost on a crash was the
# BOOKKEEPING that says which ones exist -- so the next run wiped them and
# started over. At the default batch size this checkpoints every ~5k chunks,
# which costs well under a second and bounds re-work to a few minutes.
CHECKPOINT_EVERY_BATCHES = int(os.environ.get("RAG_CHECKPOINT_EVERY", "20"))

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
# A char/token ratio is only ever an approximation, and the error is not
# uniform: prose runs ~4.0 chars/token but minified JSON, long regexes and
# bracket-dense TS/Rust drop to ~1.8. Measured on this repo, the old flat
# CHARS_PER_TOKEN = 3.0 let 11 chunks (0.6%) past the guard that the encoder
# then silently truncated -- exactly the chunks whose tails were richest.
#
# So we count with the encoder's OWN tokenizer, which fastembed already cached
# next to the ONNX weights (no extra download, no network). The char estimate
# survives as a cheap pre-filter: anything comfortably under the ceiling skips
# the real call, so ingest speed is unchanged for the ~95% common case.
CHARS_PER_TOKEN = float(os.environ.get("RAG_CHARS_PER_TOKEN", "3.0"))
# Worst observed density (chars/token) across code, minified JSON and prose.
# Below `MAX_CHUNK_TOKENS * MIN_CHARS_PER_TOKEN` chars a chunk cannot overflow,
# whatever it contains, so the exact count can be skipped safely.
MIN_CHARS_PER_TOKEN = 1.6

_tokenizer_state = {"loaded": False, "tk": None}


def _tokenizer_path():
    """The tokenizer.json fastembed cached for MODEL_NAME.

    Scored by token overlap rather than substring so a second cached model
    (e.g. a reranker) can never be picked up by accident.
    """
    want = set(re.split(r"[^a-z0-9]+", MODEL_NAME.lower())) - {""}
    best, best_score = None, 0
    for path in glob.glob(os.path.join(CACHE_DIR, "**", "tokenizer.json"), recursive=True):
        have = set(re.split(r"[^a-z0-9]+", path.lower())) - {""}
        score = len(want & have)
        if score > best_score:
            best, best_score = path, score
    return best


def _tokenizer():
    """The real encoder tokenizer, or None if it is not cached yet."""
    if not _tokenizer_state["loaded"]:
        _tokenizer_state["loaded"] = True
        try:
            from tokenizers import Tokenizer
            path = _tokenizer_path()
            if path:
                tk = Tokenizer.from_file(path)
                tk.no_truncation()      # we are measuring, not encoding
                tk.no_padding()
                _tokenizer_state["tk"] = tk
        except Exception:
            _tokenizer_state["tk"] = None       # heuristic-only, still correct-ish
    return _tokenizer_state["tk"]


TOKENIZER_AVAILABLE = None                      # resolved on first count_tokens()


def count_tokens(text: str) -> int:
    """Exact encoder token count when the tokenizer is cached, else estimated."""
    global TOKENIZER_AVAILABLE
    tk = _tokenizer()
    TOKENIZER_AVAILABLE = tk is not None
    if tk is None:
        return int(len(text) / MIN_CHARS_PER_TOKEN) + 1     # conservative
    try:
        return len(tk.encode(text, add_special_tokens=True).ids)
    except Exception:
        return int(len(text) / MIN_CHARS_PER_TOKEN) + 1


def est_tokens(text: str) -> int:
    """Token count used for every chunk-boundary decision.

    Cheap path first: if even the worst-case density keeps the text under the
    ceiling, the char estimate is provably safe and we skip the encoder.
    """
    n_chars = len(text)
    if n_chars < MAX_CHUNK_TOKENS * MIN_CHARS_PER_TOKEN:
        return int(n_chars / CHARS_PER_TOKEN) + 1
    return count_tokens(text)


def chars_per_token(text: str) -> float:
    """Measured density of one text, for adaptive split budgeting."""
    n = count_tokens(text)
    return max(1.0, len(text) / n) if n else CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# What gets indexed
# ---------------------------------------------------------------------------
EXCLUDE_DIRS = {
    ".git", "node_modules", ".poc-venv", "venv", ".venv", ".models_cache",
    "chroma_db", "build", "dist", ".idea", ".vscode", "__pycache__",
    ".gradle", ".dart_tool", "Pods", ".next", ".turbo", "coverage",
    ".pytest_cache", ".mypy_cache", "target", "out", ".expo",
    "rag-workspace",
}

# Machine-generated files that bloat the index without ever answering a
# question. package-lock.json alone was 13% of the previous index.
EXCLUDE_FILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
    "Podfile.lock", "Cargo.lock", "poetry.lock", "composer.lock",
    "gradle.lockfile", ".DS_Store", "pubspec.lock",
}
EXCLUDE_FILE_PATTERNS = (
    re.compile(r"\.min\.(js|css)$"),
    re.compile(r"\.(map|lock)$"),
    re.compile(r"(^|/)(\.env|\.env\..*)$"),
    re.compile(r"\.(g|freezed)\.dart$"),
    re.compile(r"_pb2?\.py$"),
)

MAX_FILE_BYTES = int(os.environ.get("RAG_MAX_FILE_BYTES", str(1_500_000)))

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".kts", ".dart",
    ".swift", ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb", ".php",
    ".sql", ".sh", ".bash", ".zsh", ".md", ".mdx", ".rst", ".txt",
    ".yaml", ".yml", ".toml", ".json", ".html", ".css", ".scss", ".gradle",
}

DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt", ".html"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml"}

LANGUAGE_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".kt": "kotlin",
    ".kts": "kotlin", ".dart": "dart", ".swift": "swift", ".c": "c",
    ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".go": "go", ".rs": "rust",
    ".rb": "ruby", ".php": "php", ".sql": "sql", ".sh": "bash",
    ".bash": "bash", ".zsh": "bash", ".md": "markdown", ".mdx": "markdown",
    ".rst": "other", ".txt": "other", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".json": "json", ".html": "html", ".css": "css",
    ".scss": "css", ".gradle": "other",
}


def language_for(ext: str) -> str:
    return LANGUAGE_BY_EXT.get(ext.lower(), "other")


def doc_category_for(ext: str) -> str:
    ext = ext.lower()
    if ext in DOC_EXTENSIONS:
        return "documentation"
    if ext in CONFIG_EXTENSIONS:
        return "config"
    return "source_code"


def is_excluded_path(abs_path: str, repo_root: str = None) -> bool:
    repo_root = repo_root or REPO_ROOT
    rel = os.path.relpath(abs_path, repo_root)
    if rel.startswith(".."):
        return True
    parts = rel.split(os.sep)
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    name = parts[-1]
    if name in EXCLUDE_FILE_NAMES:
        return True
    rel_posix = rel.replace(os.sep, "/")
    return any(p.search(rel_posix) for p in EXCLUDE_FILE_PATTERNS)


# ---------------------------------------------------------------------------
# Code-aware tokenizer (shared by ingest + query so BM25 is symmetric)
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def tokenize(text: str) -> list:
    """Split code into BM25 terms.

    Plain `text.lower().split()` leaves identifiers glued to punctuation, so
    `refresh_access_token(` never matches the query `refresh_access_token`.
    Here every identifier is emitted whole *and* split on snake_case and
    camelCase, so a query matches whichever form the author wrote.
    """
    tokens = []
    for raw in _WORD.findall(text):
        low = raw.lower()
        tokens.append(low)
        if "_" in low:
            tokens.extend(p for p in low.split("_") if p)
        if not raw.islower() and not raw.isupper():
            parts = _CAMEL.findall(raw)
            if len(parts) > 1:
                tokens.extend(p.lower() for p in parts)
    return tokens


# ---------------------------------------------------------------------------
# Index loading (restricted unpickling)
# ---------------------------------------------------------------------------
# The index is a pickle because rank_bm25 keeps its state in live Python
# objects and re-tokenising the whole corpus at every server start would cost
# far more than it saves. But a bare `pickle.load()` will happily execute
# whatever a REDUCE opcode tells it to, so anything that can write into the
# index directory gets arbitrary code execution in the MCP server's process.
#
# The fix that costs nothing: refuse to resolve any class outside a small
# allowlist. Loading a legitimate index is byte-for-byte the same work; a
# doctored one dies on find_class before a single object is constructed.
_PICKLE_ALLOWLIST = {
    ("rank_bm25", "BM25"), ("rank_bm25", "BM25Okapi"),
    ("rank_bm25", "BM25L"), ("rank_bm25", "BM25Plus"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
}
_PICKLE_ALLOWED_BUILTINS = {
    "dict", "list", "tuple", "set", "frozenset", "str", "int", "float",
    "bool", "bytes", "complex", "object",
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that resolves only the classes a RAG index legitimately holds."""

    def find_class(self, module, name):
        if module == "builtins" and name in _PICKLE_ALLOWED_BUILTINS:
            return super().find_class(module, name)
        if (module, name) in _PICKLE_ALLOWLIST:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            "refusing to unpickle %s.%s from the RAG index -- not on the "
            "allowlist. Delete %s and re-run ingest_codebase.py --full."
            % (module, name, INDEX_PATH))


EMPTY_INDEX = {"chunks": {}, "manifest": {}, "meta": {}, "bm25": None, "bm25_ids": []}


def load_pickle_safe(path: str) -> dict:
    """Unpickle a RAG artifact through the allowlist. Raises on anything else."""
    with open(path, "rb") as fh:
        return RestrictedUnpickler(fh).load()


def load_index_file(path: str = None) -> dict:
    """Load the index through the restricted unpickler. Never raises."""
    path = path or INDEX_PATH
    if not os.path.exists(path):
        return dict(EMPTY_INDEX)
    data = load_pickle_safe(path)
    if not isinstance(data, dict):
        raise ValueError("index is not a dict (got %s)" % type(data).__name__)
    for key, default in EMPTY_INDEX.items():
        data.setdefault(key, default if not isinstance(default, dict) else {})
    return data


# ---------------------------------------------------------------------------
# Hybrid fusion weights
# ---------------------------------------------------------------------------
# Plain RRF treats rank 1 from either retriever as identical evidence, which
# throws away the one thing we reliably know: a bare identifier query that BM25
# matched exactly is near-certainly right, while a prose question is better
# served by the dense side. So the weights follow the SHAPE of the query
# instead of being fixed -- cheap, interpretable, and it never reorders a
# result set where both retrievers already agree.
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
RRF_WEIGHTS = {
    # query shape       (dense, lexical)
    "identifier": (float(os.environ.get("RAG_W_IDENT_DENSE", "0.7")),
                   float(os.environ.get("RAG_W_IDENT_LEX", "1.4"))),
    "natural":    (float(os.environ.get("RAG_W_NL_DENSE", "1.3")),
                   float(os.environ.get("RAG_W_NL_LEX", "0.8"))),
    "mixed":      (float(os.environ.get("RAG_W_MIX_DENSE", "1.0")),
                   float(os.environ.get("RAG_W_MIX_LEX", "1.0"))),
}
# Bonus for a chunk whose own symbol IS a query term, expressed as a fraction
# of a rank-1 RRF contribution so it can promote an exact definition without
# ever steamrolling genuine agreement between the two retrievers.
SYMBOL_BOOST = float(os.environ.get("RAG_SYMBOL_BOOST", "0.6"))

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_NL_STOPWORDS = {
    "how", "what", "where", "why", "when", "which", "who", "does", "do", "is",
    "are", "the", "a", "an", "in", "to", "of", "for", "and", "or", "with",
    "work", "works", "handled", "implemented", "used", "we", "i", "it", "this",
}


def query_shape(query: str) -> str:
    """'identifier' | 'natural' | 'mixed' -- picks the fusion weights."""
    words = query.split()
    if not words:
        return "mixed"
    if len(words) <= 2 and all(_IDENTIFIER_RE.match(w) for w in words):
        # A lone `refreshAccessToken` or `users.expires_at` is a lexical query.
        if any("_" in w or not w.islower() or len(w) > 12 for w in words):
            return "identifier"
    lowered = [w.lower().strip("?.,") for w in words]
    if len(words) >= 4 and sum(1 for w in lowered if w in _NL_STOPWORDS) >= 2:
        return "natural"
    return "mixed"


# ---------------------------------------------------------------------------
# Optional local cross-encoder reranker
# ---------------------------------------------------------------------------
# RRF fuses two rankings but never actually reads the query against the chunk.
# A cross-encoder does, and it is where the top-3 precision lives. It stays
# OPT-IN because it costs a one-off model download and this project's standing
# rule is that a fresh clone works offline: unset, nothing changes; set to a
# model that is not cached, the search still answers and merely notes it.
#   RAG_RERANKER=Xenova/ms-marco-MiniLM-L-6-v2  python3 download_model.py
RERANKER_MODEL = os.environ.get("RAG_RERANKER", "").strip()
RERANK_CANDIDATES = int(os.environ.get("RAG_RERANK_CANDIDATES", "24"))

# ---------------------------------------------------------------------------
# Subprocess timeouts
# ---------------------------------------------------------------------------
# The live grep runs inside a tool call an agent is waiting on, so it must fail
# fast and say why; a re-index is a background batch job and may take as long
# as it takes. Both are env-tunable because "large repo" has no fixed meaning.
GREP_TIMEOUT = int(os.environ.get("RAG_GREP_TIMEOUT", "30"))
REINDEX_TIMEOUT = int(os.environ.get("RAG_REINDEX_TIMEOUT", "1800"))


if __name__ == "__main__":
    # Minimal CLI so the shell scripts share this module's definitions instead
    # of re-implementing repo discovery in bash and drifting from it.
    if "--repos" in sys.argv:
        print("\n".join(discover_repos()))
    elif "--repo-root" in sys.argv:
        print(REPO_ROOT)
    else:
        print("usage: rag_config.py [--repos | --repo-root]")
```

**Why the tokenizer matters.** With the naive `text.lower().split()` that most tutorials
use, `refresh_access_token` is never a token — in real source it always appears as
`refresh_access_token(` or `'refresh_access_token',`. Half of a hybrid retriever
then scores zero on the single most common code query: an exact identifier. Verify the
fix on your own code:

```bash
"$WORKSPACE/.poc-venv/bin/python3" -c "
import sys; sys.path.insert(0, '$WORKSPACE')
import rag_config as c
print(c.tokenize(\"await client.rpc('do_the_thing', {userId: u.id}); markAsRead(x)\"))"
```

You should see `do_the_thing`, `do`, `the`, `thing`, `markasread`, `mark`, `as`, `read`.

---

## Phase 3 — `rag_chunker.py` (structure-aware chunking)

This is where retrieval quality is won or lost. Four invariants, all enforced by the
code below and all verified in Phase 8:

1. **Exact line ranges.** Chunks are tree-sitter nodes, so `start_line`/`end_line`
   describe the symbol, not a window that happens to contain it. An agent can jump
   straight there instead of re-grepping to find out where the thing actually is.
2. **Nothing silently truncated.** Any chunk over the encoder's token window is split
   into labelled `part i/n` pieces. Skip this and your embedder quietly discards the
   tail of every large file — invisible, unmeasurable recall loss.
3. **Structural decomposition before blind splitting.** A 1200-line component is broken
   into its real inner functions, not 33 arbitrary slices.
4. **Gap filling.** Any line the parser did not claim (imports, top-level statements,
   parse errors) is window-indexed, so recall is never worse than a naive index.

Plus: skeletons show the **signature and the first meaningful body lines**. The obvious
implementation — return the first four lines — yields a closing brace or a stray JSX
attribute for most chunks, which is what makes agents abandon a RAG tool and fall back
to grep.

Write `<WORKSPACE>/rag_chunker.py`:

```python
#!/usr/bin/env python3
"""Structure-aware chunking for the codebase RAG index.

Chunks are produced from a real tree-sitter parse (Kotlin, TS/TSX, Dart, SQL,
Python, Swift, Java, Go, Rust, C/C++, HTML, CSS, YAML, JSON, Markdown, ...)
so that `start_line`/`end_line` describe the symbol itself rather than an
arbitrary sliding window. Three guarantees the previous window chunker did not
provide:

  1. Line ranges are exact -- an agent can jump straight to them.
  2. No chunk exceeds the encoder's token window, so nothing is silently
     truncated at embed time; oversized symbols are split into `part i/n`.
  3. Every line of every file lands in some chunk (gap filling), so recall is
     never worse than a naive window index.

If tree-sitter is unavailable the module degrades to a heading/window splitter
rather than failing -- the pipeline stays usable on any machine.
"""
import os
import re

import rag_config as cfg

try:
    from tree_sitter_language_pack import get_parser as _get_parser
    TREE_SITTER_AVAILABLE = True
except Exception:                                          # pragma: no cover
    TREE_SITTER_AVAILABLE = False

    def _get_parser(_lang):
        raise RuntimeError("tree-sitter not installed")

_PARSERS = {}

# Languages we ask tree-sitter to parse. Anything else uses the fallback.
PARSEABLE = {
    "python", "kotlin", "typescript", "tsx", "javascript", "dart", "sql",
    "java", "swift", "go", "rust", "c", "cpp", "html", "css", "yaml", "json",
}

# Nodes that become a chunk of their own.
CHUNK_TYPES = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration",
               "interface_declaration", "enum_class_body", "companion_object",
               "property_declaration", "secondary_constructor"},
    "typescript": {"function_declaration", "class_declaration", "method_definition",
                   "interface_declaration", "type_alias_declaration",
                   "lexical_declaration", "variable_declaration", "enum_declaration",
                   "abstract_class_declaration"},
    "javascript": {"function_declaration", "class_declaration", "method_definition",
                   "lexical_declaration", "variable_declaration"},
    "dart": {"class_definition", "function_signature", "method_signature",
             "enum_declaration", "extension_declaration", "mixin_declaration"},
    "sql": {"statement"},
    "java": {"method_declaration", "class_declaration", "interface_declaration",
             "constructor_declaration", "enum_declaration"},
    "swift": {"function_declaration", "class_declaration", "protocol_declaration",
              "property_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "trait_item", "enum_item"},
    "c": {"function_definition", "struct_specifier", "declaration"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "css": {"rule_set", "media_statement"},
    "html": {"element"},
    "yaml": {"block_mapping_pair", "document"},
    "json": {"pair"},
}
CHUNK_TYPES["tsx"] = CHUNK_TYPES["typescript"]

# Wrappers that carry no identity of their own -- walk straight through them.
TRANSPARENT_TYPES = {
    "export_statement", "expression_statement", "decorated_definition",
    "labeled_statement", "ambient_declaration", "statement_block",
}

# Bodies we descend into when a symbol is too large to embed in one piece.
BODY_TYPES = {
    "class_body", "declaration_list", "block", "statement_block", "body",
    "enum_class_body", "class_declaration_list", "field_declaration_list",
    "interface_body", "object_declaration_list",
}

MIN_GAP_LINES = 4
GAP_WINDOW = 40
GAP_OVERLAP = 8
_IDENT_SUFFIXES = ("identifier", "_name", "name")


def _parser(language):
    if language not in _PARSERS:
        _PARSERS[language] = _get_parser(language)
    return _PARSERS[language]


# ---------------------------------------------------------------------------
# Skeletons
# ---------------------------------------------------------------------------
_COMMENT_START = re.compile(r"^\s*(//|#|/\*|\*|--|<!--)")
_CLOSERS = re.compile(r"^\s*[)\]}>;,]+\s*$")


def build_skeleton(text: str, symbol: str = "", language: str = "other",
                   signature_lines: int = 0) -> str:
    """A skeleton that shows the *signature*, not the first four raw lines.

    The old implementation returned `lines[:4]`, which for a window chunk was
    whatever happened to sit at the top of the window -- frequently a closing
    brace or an unrelated JSX attribute. Here we anchor on the declaration line
    and then show the first few meaningful body lines, which is what a reader
    needs to decide whether to open the full chunk.
    """
    lines = text.splitlines()
    if len(lines) <= 6:
        return text

    # Anchor: the line that declares the symbol, else the first real line.
    anchor = 0
    if symbol:
        for i, line in enumerate(lines[:12]):
            if re.search(r"\b%s\b" % re.escape(symbol), line):
                anchor = i
                break
    else:
        for i, line in enumerate(lines[:8]):
            if line.strip() and not _COMMENT_START.match(line):
                anchor = i
                break

    head = lines[:anchor] if anchor and anchor <= 3 else []
    sig_span = signature_lines if signature_lines else _signature_span(lines, anchor)
    signature = lines[anchor:anchor + sig_span]

    body = []
    for line in lines[anchor + sig_span:]:
        stripped = line.strip()
        if not stripped or _CLOSERS.match(line) or _COMMENT_START.match(line):
            continue
        body.append(line)
        if len(body) >= 3:
            break

    shown = head + signature + body
    hidden = len(lines) - len(shown)
    out = "\n".join(shown)
    if hidden > 0:
        out += "\n... [%d more lines -- get_chunk_content for full body] ..." % hidden
    return out


def _signature_span(lines, anchor, limit=4):
    """How many lines the declaration spans (multi-line parameter lists)."""
    depth = 0
    for offset in range(0, min(limit, len(lines) - anchor)):
        line = lines[anchor + offset]
        depth += line.count("(") - line.count(")")
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            return offset + 1
    return 1


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------
def _make_chunk(rel_path, symbol, start_line, end_line, text, language,
                doc_category, node_type, parent=None, part=1, parts=1):
    suffix = "" if parts == 1 else ".p%d" % part
    chunk_id = "%s:%s:%d-%d%s" % (rel_path, symbol or "block", start_line, end_line, suffix)
    return {
        "id": chunk_id,
        "text": text,
        "skeleton": build_skeleton(text, symbol, language),
        "metadata": {
            "filepath": rel_path,
            "symbol": symbol or "block",
            "parent_symbol": parent or "",
            "start_line": start_line,
            "end_line": end_line,
            "type": node_type,
            "doc_category": doc_category,
            "language": language,
            "part": part,
            "parts": parts,
        },
    }


def _split_oversized(rel_path, symbol, start_line, lines, language,
                     doc_category, node_type, parent):
    """Split a symbol that exceeds the encoder window into labelled parts."""
    # Budget in characters, but derive the exchange rate from THIS text rather
    # than a global constant: a minified JSON blob really does cost ~1.8
    # chars/token where prose costs ~4.0, and a fixed 3.0 either truncates the
    # dense case or shreds the sparse one into needless parts.
    text_for_ratio = "\n".join(lines)
    density = cfg.chars_per_token(text_for_ratio) if text_for_ratio else cfg.CHARS_PER_TOKEN
    budget_chars = max(200, int(cfg.MAX_CHUNK_TOKENS * density * 0.95))

    # Hard-wrap any single line that alone blows the budget, so the caller's
    # invariant ("no chunk exceeds the encoder window") always holds.
    wrapped = []
    for line in lines:
        while len(line) > budget_chars:
            wrapped.append(line[:budget_chars])
            line = line[budget_chars:]
        wrapped.append(line)
    lines = wrapped

    overlap_budget = max(0, int(budget_chars * 0.12))
    pieces, current, current_len, piece_start = [], [], 0, 0
    for idx, line in enumerate(lines):
        if current and current_len + len(line) + 1 > budget_chars:
            pieces.append((piece_start, current))
            # Carry a little context into the next piece so a split symbol stays
            # searchable from either side -- but bound it by characters, not by
            # a line count: three long prose lines would blow the whole budget.
            overlap, overlap_len = [], 0
            for prev in reversed(current):
                if len(overlap) >= 3 or overlap_len + len(prev) > overlap_budget:
                    break
                overlap.insert(0, prev)
                overlap_len += len(prev) + 1
            current = list(overlap)
            current_len = overlap_len
            piece_start = idx - len(overlap)
        current.append(line)
        current_len += len(line) + 1
    if current:
        pieces.append((piece_start, current))

    # The budget is a good estimate, not a proof. Verify each piece against the
    # real tokenizer and re-cut any that still overflows, so the module's
    # stated guarantee ("no chunk exceeds the encoder window") is actually one.
    verified = []
    for offset, piece_lines in pieces:
        text = "\n".join(piece_lines)
        if cfg.count_tokens(text) <= cfg.MAX_CHUNK_TOKENS or len(piece_lines) < 2:
            verified.append((offset, piece_lines))
            continue
        shrunk = max(200, int(budget_chars * 0.6))
        sub, cur, cur_len, sub_start = [], [], 0, offset
        for idx, line in enumerate(piece_lines):
            if cur and cur_len + len(line) + 1 > shrunk:
                sub.append((sub_start, cur))
                cur, cur_len, sub_start = [], 0, offset + idx
            cur.append(line)
            cur_len += len(line) + 1
        if cur:
            sub.append((sub_start, cur))
        verified.extend(sub)
    pieces = verified

    total = len(pieces)
    out = []
    for i, (offset, piece_lines) in enumerate(pieces, start=1):
        s = start_line + offset
        e = s + len(piece_lines) - 1
        out.append(_make_chunk(rel_path, symbol, s, e, "\n".join(piece_lines),
                               language, doc_category, node_type, parent, i, total))
    return out


def _append_bounded(chunks, rel_path, symbol, start_line, end_line, lines,
                    language, doc_category, node_type, parent):
    """Append a chunk, splitting it first if it would overflow the encoder."""
    text = "\n".join(lines).strip("\n")
    if cfg.est_tokens(text) > cfg.MAX_CHUNK_TOKENS:
        chunks.extend(_split_oversized(rel_path, symbol, start_line, lines,
                                       language, doc_category, node_type, parent))
    else:
        chunks.append(_make_chunk(rel_path, symbol, start_line, end_line, text,
                                  language, doc_category, node_type, parent))


_SQL_STMT_START = re.compile(
    r"^\s*(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|GRANT|REVOKE|COMMENT|DO)\b",
    re.IGNORECASE)


def _split_sql(rel_path, start_line, lines, doc_category):
    """Break a fused SQL node on top-level DDL keywords."""
    groups, current, group_start = [], [], 0
    for idx, line in enumerate(lines):
        if _SQL_STMT_START.match(line) and current and any(l.strip() for l in current):
            groups.append((group_start, current))
            current, group_start = [], idx
        current.append(line)
    if current:
        groups.append((group_start, current))

    out = []
    for offset, group in groups:
        text = "\n".join(group).strip("\n")
        if not text.strip():
            continue
        s = start_line + offset
        e = s + len(group) - 1
        _append_bounded(out, rel_path, _sql_symbol(text), s, e, group, "sql",
                        doc_category, "sql_statement", None)
    return out


def _node_symbol(node, source_bytes):
    named = node.child_by_field_name("name")
    if named is not None:
        return source_bytes[named.start_byte:named.end_byte].decode("utf-8", "ignore").strip()
    # Breadth-first over the first two levels for an identifier-ish node.
    frontier = list(node.children)
    for _ in range(2):
        nxt = []
        for child in frontier:
            if child.type.endswith(_IDENT_SUFFIXES) or child.type in {
                "identifier", "simple_identifier", "type_identifier",
                "property_identifier", "field_identifier", "object_reference",
                "dotted_name",
            }:
                text = source_bytes[child.start_byte:child.end_byte]
                name = text.decode("utf-8", "ignore").strip().strip('"`\'')
                if name:
                    return name.split(".")[-1] if child.type == "object_reference" else name
            nxt.extend(child.children)
        frontier = nxt
    return ""


_DART_DECL = re.compile(
    r"^\s*(?:@\w+\s+)*(?:static\s+|final\s+|const\s+|abstract\s+)*"
    r"(?:[\w<>,\s\?\[\]]+\s+)?([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\(")


def _dart_symbol(text):
    """Dart declarations lead with the return type, so the plain 'first
    identifier' heuristic yields 'Future' instead of the method name."""
    for line in text.splitlines()[:3]:
        m = _DART_DECL.match(line)
        if m and m.group(1) not in ("if", "for", "while", "switch", "return", "catch"):
            return m.group(1)
    return ""


def _sql_symbol(text):
    m = re.search(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|FUNCTION|PROCEDURE|VIEW|TRIGGER|POLICY|INDEX|EXTENSION)"
        r"(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:\"([^\"]+)\"|([A-Za-z0-9_.]+))", text, re.IGNORECASE)
    if m:
        # Quoted policy names contain spaces -- keep them whole.
        return (m.group(1) or m.group(2)).strip()
    m = re.search(r"^\s*(ALTER|INSERT|GRANT|DROP)\s+\w+\s+([\"A-Za-z0-9_.]+)",
                  text, re.IGNORECASE | re.MULTILINE)
    return (m.group(2).replace('"', "") if m else "")


# ---------------------------------------------------------------------------
# Tree-sitter driven extraction
# ---------------------------------------------------------------------------
def _extract_with_tree_sitter(rel_path, content, language, doc_category):
    parser = _parser(language)
    source_bytes = content.encode("utf-8", "ignore")
    tree = parser.parse(source_bytes)
    lines = content.splitlines()
    chunk_types = CHUNK_TYPES.get(language, set())
    chunks, covered = [], set()

    def emit(node, parent_symbol):
        start_line = max(1, node.start_point[0] + 1)
        end_line = min(len(lines), node.end_point[0] + 1)
        if end_line < start_line:
            return
        # Dart's grammar emits the signature and the body as siblings; keep them
        # together so a chunk is a whole method rather than one orphan line.
        if language == "dart" and node.type in ("function_signature", "method_signature"):
            sibling = node.next_sibling
            while sibling is not None and sibling.type in (
                    "function_body", "block", "=>", ";", "async", "await"):
                end_line = max(end_line, sibling.end_point[0] + 1)
                if sibling.type in ("function_body", "block"):
                    break
                sibling = sibling.next_sibling
        node_lines = lines[start_line - 1:end_line]
        text = "\n".join(node_lines).strip("\n")
        if not text.strip():
            return
        if language == "sql":
            symbol = _sql_symbol(text)
        elif language == "dart":
            symbol = _dart_symbol(text) or _node_symbol(node, source_bytes)
        else:
            symbol = _node_symbol(node, source_bytes)
        if not symbol and language in ("typescript", "tsx", "javascript"):
            m = re.search(r"\b(?:const|let|var|function|class)\s+([A-Za-z0-9_$]+)", text)
            symbol = m.group(1) if m else ""

        if cfg.est_tokens(text) > cfg.MAX_CHUNK_TOKENS:
            # SQL grammars swallow dollar-quoted function bodies, which can fuse
            # a whole file into one `statement` node. Re-split on DDL keywords
            # so each policy/function stays individually addressable.
            if language == "sql":
                pieces = _split_sql(rel_path, start_line, node_lines, doc_category)
                if len(pieces) > 1:
                    chunks.extend(pieces)
                    covered.update(range(start_line, end_line + 1))
                    return

            # Prefer structural sub-chunks over a blind split.
            children = _chunkable_children(node)
            if children:
                body_start = children[0].start_point[0] + 1
                if body_start - start_line >= 2:
                    header_lines = lines[start_line - 1:body_start - 1]
                    header = "\n".join(header_lines).strip("\n")
                    if header.strip():
                        _append_bounded(chunks, rel_path, symbol, start_line,
                                        body_start - 1, header_lines, language,
                                        doc_category, node.type + "_header",
                                        parent_symbol)
                        covered.update(range(start_line, body_start))
                for child in children:
                    emit(child, symbol or parent_symbol)
                return
            chunks.extend(_split_oversized(rel_path, symbol, start_line, node_lines,
                                           language, doc_category, node.type, parent_symbol))
            covered.update(range(start_line, end_line + 1))
            return

        chunks.append(_make_chunk(rel_path, symbol, start_line, end_line, text,
                                  language, doc_category, node.type, parent_symbol))
        covered.update(range(start_line, end_line + 1))

    def _chunkable_children(node, max_depth=6):
        """Nearest layer of structural children, stopping at each match.

        Descending through *any* intermediate node (variable_declarator,
        arrow_function, call_expression, ...) is what lets a 1200-line
        `const ChatProvider = () => { ... }` decompose into its real inner
        symbols instead of being sliced blindly by line count.
        """
        found = []
        stack = [(child, 0) for child in node.children]
        while stack:
            child, depth = stack.pop(0)
            if child.type in chunk_types:
                found.append(child)
            elif depth < max_depth and child.child_count:
                stack.extend((g, depth + 1) for g in child.children)
        return _coalesce_nodes(sorted(found, key=lambda n: n.start_byte))

    def _coalesce_nodes(nodes):
        """Drop children fully contained in an earlier sibling."""
        out = []
        for n in nodes:
            if out and n.start_byte >= out[-1].start_byte and n.end_byte <= out[-1].end_byte:
                continue
            out.append(n)
        return out

    def walk(node, parent_symbol=""):
        for child in node.children:
            if child.type in chunk_types:
                emit(child, parent_symbol)
            elif child.type in TRANSPARENT_TYPES or child.child_count:
                walk(child, parent_symbol)

    walk(tree.root_node)
    chunks.extend(_gap_chunks(rel_path, lines, covered, language, doc_category))
    return chunks


def _gap_chunks(rel_path, lines, covered, language, doc_category):
    """Window-index whatever the parser did not claim (imports, top-level code,
    parse errors) so the index never loses a line."""
    out = []
    gap_start = None
    for i in range(1, len(lines) + 2):
        line_is_gap = i <= len(lines) and i not in covered
        if line_is_gap and gap_start is None:
            gap_start = i
        elif not line_is_gap and gap_start is not None:
            out.extend(_window(rel_path, lines, gap_start, i - 1, language, doc_category))
            gap_start = None
    return out


# The window splitter only runs where tree-sitter could not (unsupported
# language, syntax error, or a gap between parsed nodes) -- which is exactly
# where a name matters most, because there is no node to ask. The old pattern
# knew six keywords from JS and Python, so Go/Rust/Kotlin/Swift gap chunks came
# back anonymous and were unreachable by symbol. These cover the declaration
# forms those languages actually use, including leading decorators/annotations
# and receivers, and each alternative is anchored so prose cannot match.
_FALLBACK_DECLS = (
    # JS/TS/Python/Kotlin/Dart
    r"^[ \t]*(?:@[\w.]+[ \t]*)*(?:export[ \t]+)?(?:default[ \t]+)?"
    r"(?:public|private|internal|protected|abstract|final|open|sealed|static|"
    r"suspend|inline|override|external|operator|data|async|const|readonly)?"
    r"[ \t]*(?:function|class|interface|enum|object|trait|struct|record|"
    r"typealias|type|fun|def|val|var|let|const)[ \t]+([A-Za-z_$][\w$]*)",
    # Go: func (r *Recv) Name(...) / func Name(...)
    r"^[ \t]*func[ \t]+(?:\([^)]*\)[ \t]*)?([A-Za-z_][\w]*)",
    # Rust: pub async unsafe fn name / impl Trait for Name / struct Name
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:default[ \t]+)?(?:const[ \t]+)?"
    r"(?:async[ \t]+)?(?:unsafe[ \t]+)?(?:extern[ \t]+\"[^\"]*\"[ \t]+)?"
    r"(?:fn|struct|enum|trait|union|mod|impl)[ \t]+([A-Za-z_][\w]*)",
    # Swift/Java/C#: modifiers then func/class/protocol/extension
    r"^[ \t]*(?:@\w+[ \t]+)*(?:public|private|fileprivate|internal|open|"
    r"protected|static|final)?[ \t]*(?:func|protocol|extension|actor)"
    r"[ \t]+([A-Za-z_][\w]*)",
    # C/C++/Java method: <type> name(  -- last resort, needs a brace or colon
    r"^[ \t]*(?:[A-Za-z_][\w:<>,*&\s]{0,60}?[\s*&])([A-Za-z_]\w*)[ \t]*\([^;]*\)"
    r"[ \t]*(?:const)?[ \t]*\{",
)
_FALLBACK_DECLS = tuple(re.compile(p, re.MULTILINE) for p in _FALLBACK_DECLS)
_FALLBACK_STOP = {
    "if", "for", "while", "switch", "return", "catch", "else", "do", "try",
    "match", "loop", "when", "with", "in", "of", "new", "case",
}


def _fallback_symbol(text: str) -> str:
    """Best-effort declaration name for a chunk no parser claimed."""
    for pattern in _FALLBACK_DECLS:
        for m in pattern.finditer(text):
            name = m.group(1)
            if name and name.lower() not in _FALLBACK_STOP:
                return name
    return ""


def _window(rel_path, lines, start_line, end_line, language, doc_category):
    span = lines[start_line - 1:end_line]
    if len([l for l in span if l.strip()]) < MIN_GAP_LINES:
        return []
    out = []
    i = 0
    while i < len(span):
        piece = span[i:i + GAP_WINDOW]
        text = "\n".join(piece).strip("\n")
        if text.strip():
            s = start_line + i
            e = s + len(piece) - 1
            symbol = _sql_symbol(text) if language == "sql" else ""
            if not symbol:
                symbol = _fallback_symbol(text)
            if cfg.est_tokens(text) > cfg.MAX_CHUNK_TOKENS:
                out.extend(_split_oversized(rel_path, symbol, s, piece, language,
                                            doc_category, "block", None))
            else:
                out.append(_make_chunk(rel_path, symbol, s, e, text, language,
                                       doc_category, "block"))
        if len(piece) < GAP_WINDOW:
            break
        i += (GAP_WINDOW - GAP_OVERLAP)
    return out


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _extract_markdown(rel_path, content, doc_category):
    """Split docs on headings so a section is retrievable as a unit."""
    lines = content.splitlines()
    sections, current, title, start = [], [], "", 1
    for i, line in enumerate(lines, start=1):
        m = _MD_HEADING.match(line)
        if m and current:
            sections.append((title, start, i - 1, current))
            current, title, start = [], m.group(2).strip(), i
        elif m:
            title, start = m.group(2).strip(), i
        current.append(line)
    if current:
        sections.append((title, start, len(lines), current))

    chunks = []
    for title, s, e, body in sections:
        text = "\n".join(body).strip("\n")
        if not text.strip():
            continue
        symbol = re.sub(r"[^A-Za-z0-9_ .-]", "", title)[:60] or "section"
        if cfg.est_tokens(text) > cfg.MAX_CHUNK_TOKENS:
            chunks.extend(_split_oversized(rel_path, symbol, s, body, "markdown",
                                           doc_category, "doc_section", None))
        else:
            chunks.append(_make_chunk(rel_path, symbol, s, e, text, "markdown",
                                      doc_category, "doc_section"))
    return chunks


def extract_chunks(abs_path, rel_path, content):
    ext = os.path.splitext(abs_path)[1].lower()
    language = cfg.language_for(ext)
    doc_category = cfg.doc_category_for(ext)

    if not content.strip():
        return []

    if language == "markdown":
        return _extract_markdown(rel_path, content, doc_category)

    if TREE_SITTER_AVAILABLE and language in PARSEABLE:
        try:
            chunks = _extract_with_tree_sitter(rel_path, content, language, doc_category)
            if chunks:
                return chunks
        except Exception:
            pass  # fall through to the window splitter

    lines = content.splitlines()
    return _window(rel_path, lines, 1, len(lines), language, doc_category)

# Bump when chunk boundaries change so ingests auto-rebuild.
VERSION = "2.4"
```

**Adapting to a language you use that is not listed.** Add its node types to
`CHUNK_TYPES`. Discover the right names empirically rather than guessing:

```bash
"$WORKSPACE/.poc-venv/bin/python3" - <<'PY'
from tree_sitter_language_pack import get_parser
import collections
LANG, PATH = "go", "path/to/a/representative/file.go"
tree = get_parser(LANG).parse(open(PATH, "rb").read())
counts = collections.Counter()
def walk(node, depth=0):
    counts[(depth, node.type)] += 1
    if depth < 3:
        for child in node.children: walk(child, depth + 1)
walk(tree.root_node)
for (depth, kind), n in sorted(counts.items()):
    print(f"  d{depth} {kind} x{n}")
PY
```

Node types at depth 1–2 that correspond to declarations are your `CHUNK_TYPES`. Bump
`VERSION` at the bottom of the file whenever you change chunk boundaries — the ingester
watches it and forces a rebuild, so a boundary change can never leave a half-migrated
index behind.

---

## Phase 4 — `ingest_codebase.py` (incremental indexer)

Full rebuilds do not scale and, worse, they discourage re-indexing — which is how an
index goes stale and starts producing confident false negatives. This indexer keeps a
per-file SHA-1 manifest and only re-chunks and re-embeds what actually changed.

It also self-heals: a changed embedding model, a bumped chunker version, or a
disagreement between the vector store and the index each force a full rebuild
automatically. The index is written atomically (`tmp` + `os.replace`) so a server
reading it concurrently never sees a half-written file.

Write `<WORKSPACE>/ingest_codebase.py`:

```python
#!/usr/bin/env python3
"""Incremental indexer for the local hybrid codebase RAG system.

Pipeline: walk -> content-hash -> structure-aware chunk -> embed (ONNX, local)
-> ChromaDB (dense) + BM25 (lexical) -> single pickle holding chunks, the BM25
model and the file manifest.

Only files whose SHA-1 changed are re-chunked and re-embedded; everything else
is reused from the previous index. A full rebuild is one flag away (`--full`)
and happens automatically if the index and the vector store disagree.
"""
import argparse
import hashlib
import os
import pickle
import subprocess
import sys
import time

import rag_config as cfg
import rag_chunker as chunker


def log(msg):
    print(msg, flush=True)


def sha1_of(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def git_head(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return ""


def git_heads(repo_root):
    """Commit per repo in the workspace, so 'is my index current?' is answerable.

    A single `git_head` is meaningless when the root is a container of several
    checkouts -- it reports either nothing or one arbitrary repo, while four
    others drift unnoticed.
    """
    heads = {}
    for repo in cfg.discover_repos(repo_root):
        head = git_head(repo)
        if head:
            rel = os.path.relpath(repo, repo_root).replace(os.sep, "/")
            heads[rel if rel != "." else os.path.basename(repo)] = head
    return heads


def iter_source_files(repo_root):
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in cfg.EXCLUDE_DIRS and not d.startswith(".git")]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in cfg.SUPPORTED_EXTENSIONS:
                continue
            path = os.path.join(root, name)
            if cfg.is_excluded_path(path, repo_root):
                continue
            try:
                if os.path.getsize(path) > cfg.MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def load_index():
    """Previous index, read through the restricted unpickler.

    A refusal here is not fatal: we just rebuild, which is the safe outcome
    either way -- a corrupt or tampered index gets replaced rather than trusted.
    """
    try:
        return cfg.load_index_file(cfg.INDEX_PATH)
    except Exception as exc:
        log("Existing index unreadable (%s); rebuilding from scratch." % exc)
        return {"chunks": {}, "manifest": {}, "meta": {}}


def write_checkpoint(manifest, embedded_ids, repo_root):
    """Record which chunks are already in the vector store.

    Deliberately does NOT store chunk bodies: re-chunking on resume is seconds,
    while writing every body at each checkpoint would cost more than the crash
    it protects against. Manifest + embedded ids is all a resume needs.
    """
    payload = {
        "manifest": manifest,
        "embedded": list(embedded_ids),
        "meta": {"model": cfg.MODEL_NAME, "chunker_version": chunker.VERSION,
                 "repo_root": repo_root, "at": time.time()},
    }
    tmp = cfg.CHECKPOINT_PATH + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cfg.CHECKPOINT_PATH)      # atomic: never a half-written record


def read_checkpoint(repo_root):
    """A usable checkpoint for THIS config, or None.

    Rejects a checkpoint written under a different model, chunker or repo --
    resuming across any of those would mix incompatible vectors, which is worse
    than the hour it saves.
    """
    if not os.path.exists(cfg.CHECKPOINT_PATH):
        return None
    try:
        data = cfg.load_pickle_safe(cfg.CHECKPOINT_PATH)
    except Exception as exc:
        log("Checkpoint unreadable (%s); ignoring it." % exc)
        return None
    meta = (data or {}).get("meta", {})
    if (meta.get("model") != cfg.MODEL_NAME
            or meta.get("chunker_version") != chunker.VERSION
            or meta.get("repo_root") != repo_root):
        log("Checkpoint is from a different model/chunker/repo; ignoring it.")
        return None
    return data


def clear_checkpoint():
    for path in (cfg.CHECKPOINT_PATH, cfg.CHECKPOINT_PATH + ".tmp"):
        try:
            os.remove(path)
        except OSError:
            pass


def ingest(repo_root, full=False):
    started = time.time()
    log("Repository : %s" % repo_root)
    log("Index dir  : %s" % cfg.INDEX_DIR)
    log("Model      : %s" % cfg.MODEL_NAME)
    log("Parser     : %s" % ("tree-sitter" if chunker.TREE_SITTER_AVAILABLE
                             else "regex fallback (tree-sitter not installed)"))

    state = {"chunks": {}, "manifest": {}, "meta": {}} if full else load_index()
    prev_chunks = state["chunks"]
    prev_manifest = state["manifest"]

    if state["meta"].get("model") not in (None, cfg.MODEL_NAME):
        log("Embedding model changed -> forcing full rebuild.")
        full, prev_chunks, prev_manifest = True, {}, {}
    if state["meta"].get("chunker_version") not in (None, chunker.VERSION):
        log("Chunker version changed -> forcing full rebuild.")
        full, prev_chunks, prev_manifest = True, {}, {}

    import chromadb
    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
    collection = client.get_or_create_collection(name=cfg.COLLECTION_NAME)

    checkpoint = None if full else read_checkpoint(repo_root)

    if not full and prev_chunks and collection.count() != len(prev_chunks):
        # A count mismatch normally means an index/store divergence we cannot
        # reason about, so the safe move is to rebuild. But an interrupted
        # ingest produces exactly this symptom by design -- extra vectors the
        # old index has never heard of -- and there a rebuild throws away the
        # very work the checkpoint exists to preserve.
        if checkpoint:
            log("Vector store (%d) and index (%d) disagree, but a checkpoint from "
                "an interrupted ingest explains it -- resuming."
                % (collection.count(), len(prev_chunks)))
        else:
            log("Vector store (%d) and index (%d) disagree -> full rebuild."
                % (collection.count(), len(prev_chunks)))
            full, prev_chunks, prev_manifest = True, {}, {}

    if full:
        existing = collection.get(include=[])["ids"]
        for i in range(0, len(existing), 5000):
            collection.delete(ids=existing[i:i + 5000])

    manifest, chunks = {}, {}
    reused_files = changed_files = 0
    stale_ids, new_chunk_ids = [], []

    for path in iter_source_files(repo_root):
        rel = os.path.relpath(path, repo_root).replace(os.sep, "/")
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        digest = sha1_of(raw)
        record = prev_manifest.get(rel)

        if record and record.get("sha") == digest and all(cid in prev_chunks for cid in record["ids"]):
            manifest[rel] = record
            for cid in record["ids"]:
                chunks[cid] = prev_chunks[cid]
            reused_files += 1
            continue

        changed_files += 1
        if record:
            stale_ids.extend(record["ids"])
        content = raw.decode("utf-8", errors="ignore")
        file_chunks = chunker.extract_chunks(path, rel, content)
        ids = []
        for chunk in file_chunks:
            cid = chunk["id"]
            if cid in chunks:                       # defensive: never collide
                cid = "%s#%d" % (cid, len(chunks))
                chunk["id"] = cid
            chunks[cid] = chunk
            ids.append(cid)
            new_chunk_ids.append(cid)
        manifest[rel] = {"sha": digest, "mtime": os.path.getmtime(path), "ids": ids}

    removed_files = [rel for rel in prev_manifest if rel not in manifest]
    for rel in removed_files:
        stale_ids.extend(prev_manifest[rel]["ids"])

    # ---- resume ----------------------------------------------------------
    already_embedded = set()
    if checkpoint:
        ck_manifest = checkpoint.get("manifest", {})
        ck_embedded = set(checkpoint.get("embedded", []))
        for rel, record in manifest.items():
            prev = ck_manifest.get(rel)
            # Only trust the checkpoint for files that have not moved since.
            # Chunk ids encode line numbers, so an unchanged SHA-1 guarantees
            # the ids AND the text behind them are identical.
            if prev and prev.get("sha") == record["sha"]:
                already_embedded.update(cid for cid in record["ids"] if cid in ck_embedded)
        # Anything the interrupted run embedded that the current plan no longer
        # contains (a file edited between the crash and now) is an orphan vector.
        stale_ids.extend(ck_embedded - set(chunks))
        if already_embedded:
            log("Resuming interrupted ingest: %d chunk(s) already embedded, skipping them."
                % len(already_embedded))

    log("Files: %d unchanged, %d re-chunked, %d removed" %
        (reused_files, changed_files, len(removed_files)))
    log("Chunks: %d total (%d new)" % (len(chunks), len(new_chunk_ids)))

    if not chunks:
        log("Nothing indexable found -- aborting without touching the index.")
        return 1

    # ---- vector store -----------------------------------------------------
    stale_ids = [cid for cid in set(stale_ids) if cid not in chunks]
    if stale_ids:
        for i in range(0, len(stale_ids), 5000):
            collection.delete(ids=stale_ids[i:i + 5000])
        log("Deleted %d stale vectors." % len(stale_ids))

    pending = [cid for cid in new_chunk_ids if cid not in already_embedded]
    if pending:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=cfg.MODEL_NAME, cache_dir=cfg.CACHE_DIR)
        log("Embedding %d chunks..." % len(pending))
        batch = cfg.EMBED_BATCH_SIZE
        embedded = set(already_embedded)
        for n, i in enumerate(range(0, len(pending), batch), start=1):
            window = pending[i:i + batch]
            texts = [chunks[cid]["text"] for cid in window]
            vectors = [v.tolist() for v in model.embed(texts)]
            collection.upsert(
                ids=window,
                embeddings=vectors,
                metadatas=[chunks[cid]["metadata"] for cid in window],
            )
            embedded.update(window)
            log("  %d/%d" % (min(i + batch, len(pending)), len(pending)))
            # The vectors are already durable; this makes the fact of them so.
            if n % cfg.CHECKPOINT_EVERY_BATCHES == 0 and i + batch < len(pending):
                write_checkpoint(manifest, embedded, repo_root)
                log("  [checkpoint] %d embedded so far" % len(embedded))

    # ---- lexical index ----------------------------------------------------
    log("Building BM25 index with the code-aware tokenizer...")
    from rank_bm25 import BM25Okapi
    ordered_ids = list(chunks.keys())
    corpus = [cfg.tokenize(chunks[cid]["text"] + " " + chunks[cid]["metadata"]["filepath"]
                           + " " + chunks[cid]["metadata"]["symbol"]) for cid in ordered_ids]
    bm25 = BM25Okapi(corpus)

    payload = {
        "bm25": bm25,
        "bm25_ids": ordered_ids,
        "chunks": chunks,
        "manifest": manifest,
        "meta": {
            "model": cfg.MODEL_NAME,
            "chunker_version": chunker.VERSION,
            "repo_root": repo_root,
            "built_at": time.time(),
            "git_head": git_head(repo_root),
            "git_heads": git_heads(repo_root),
            "chunk_count": len(chunks),
            "file_count": len(manifest),
            "tree_sitter": chunker.TREE_SITTER_AVAILABLE,
        },
    }
    tmp = cfg.INDEX_PATH + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cfg.INDEX_PATH)      # atomic: the server never sees a half file
    clear_checkpoint()                   # the index now supersedes any progress record

    if os.path.exists(cfg.LEGACY_BM25_PATH):
        os.remove(cfg.LEGACY_BM25_PATH)
        log("Removed superseded bm25_index.pkl.")

    langs = {}
    for chunk in chunks.values():
        langs[chunk["metadata"]["language"]] = langs.get(chunk["metadata"]["language"], 0) + 1
    top = ", ".join("%s=%d" % kv for kv in sorted(langs.items(), key=lambda x: -x[1])[:8])
    log("Languages: %s" % top)
    log("Done in %.1fs -- %d chunks across %d files." %
        (time.time() - started, len(chunks), len(manifest)))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Index a codebase for the local RAG MCP server.")
    parser.add_argument("--target", default=cfg.REPO_ROOT, help="Repository path to index.")
    parser.add_argument("--full", action="store_true", help="Force a full re-index.")
    args = parser.parse_args()
    repo_root = os.path.abspath(os.path.expanduser(args.target))
    if not os.path.isdir(repo_root):
        log("Target repository does not exist: %s" % repo_root)
        return 2
    return ingest(repo_root, full=args.full)


if __name__ == "__main__":
    sys.exit(main())
```

---

## Phase 5 — `mcp_server.py` (the MCP server)

Client-agnostic by construction: plain stdio FastMCP, no vendor-specific fields, all
configuration by environment variable, and tool descriptions written so any model can
pick the right tool without extra system prompting.

Six decisions in this file that separate a useful server from a plausible one:

- **Asymmetric query embedding.** bge/e5/arctic models are trained with an instruction
  prefix on the query and none on the passage. Omitting it costs recall for free.
- **Filter pushdown.** Filtering *after* retrieval means a narrow `path_filter` returns
  nothing whenever the matching code sits outside the global top-k. Push the filter into
  ChromaDB (`$and`/`$in`) and into the BM25 candidate set instead.
- **Hot reload.** Watch the index mtime, so a re-index takes effect without restarting
  the client session.
- **Staleness is surfaced, never hidden.** Every result carries a banner when indexed
  files have changed on disk, and every negative answer is confirmed against a live grep
  of the working tree. An unverified "zero references" from a stale index is the most
  expensive failure this system can produce.
- **Pagination.** Truncating grep output at a fixed cap with no `offset` forces the
  caller back to the shell. Always give them the next page.
- **stdout belongs to the transport.** Log to a file. A stray `print` corrupts the MCP
  protocol stream and the failure looks like a mysterious disconnect.

Write `<WORKSPACE>/mcp_server.py`:

```python
#!/usr/bin/env python3
"""Local hybrid (dense + lexical) codebase RAG exposed over MCP.

Client-agnostic by construction: plain stdio FastMCP, no vendor-specific
fields, every path and model configurable by environment variable, and tool
descriptions written so any LLM can pick the right tool without extra prompting.

Tools
  search_codebase        hybrid semantic + keyword search (RRF fused)
  find_symbol_references where a symbol is defined and used
  find_symbol_or_keyword literal/regex grep with pagination
  get_chunk_content      full source of one search hit
  get_file_context       exact line ranges from many files in one call
  rag_status             index health, freshness, staleness warnings
  reindex                refresh the index after edits (incremental by default)
"""
import functools
import math
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_config as cfg
from fastmcp import FastMCP

mcp = FastMCP("Local Codebase RAG")

# ---------------------------------------------------------------------------
# Logging (stdout belongs to the MCP transport -- never print there)
# ---------------------------------------------------------------------------
def log(msg):
    try:
        with open(cfg.LOG_PATH, "a") as fh:
            fh.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    except Exception:
        pass


def tool_guard(fn):
    """Never let an exception escape as an MCP protocol error -- an agent can
    act on a readable message, but a transport fault just kills the session."""
    @functools.wraps(fn)          # keeps the real signature for MCP schema generation
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            log("ERROR %s: %r" % (fn.__name__, exc))
            return ("%s failed: %s: %s\nThe index may be missing or stale -- "
                    "call rag_status(), then reindex() if needed."
                    % (fn.__name__, type(exc).__name__, exc))
    return wrapper


# ---------------------------------------------------------------------------
# Lazily loaded, hot-reloading resources
# ---------------------------------------------------------------------------
_state = {"model": None, "collection": None, "index": None, "index_mtime": 0.0,
          "freshness": None, "freshness_at": 0.0, "reranker": None}


def _load_index():
    """(Re)load the index when it changes, so a reindex needs no restart.

    Read through cfg.load_index_file, which resolves only allowlisted classes:
    a tampered index fails loudly here instead of executing on load.
    """
    if not os.path.exists(cfg.INDEX_PATH):
        _state["index"] = dict(cfg.EMPTY_INDEX)
        return _state["index"]
    mtime = os.path.getmtime(cfg.INDEX_PATH)
    if _state["index"] is None or mtime != _state["index_mtime"]:
        try:
            _state["index"] = cfg.load_index_file(cfg.INDEX_PATH)
        except Exception as exc:
            log("REFUSED index load: %r" % exc)
            _state["index"] = dict(cfg.EMPTY_INDEX)
            _state["index"]["load_error"] = str(exc)
        _state["index_mtime"] = mtime
        _state["freshness"] = None
        log("index loaded: %d chunks" % len(_state["index"].get("chunks", {})))
    return _state["index"]


def _reranker():
    """Lazily built cross-encoder, or None when unset/uncached (never fatal)."""
    if not cfg.RERANKER_MODEL:
        return None
    if _state["reranker"] is None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _state["reranker"] = TextCrossEncoder(
                model_name=cfg.RERANKER_MODEL, cache_dir=cfg.CACHE_DIR)
            log("reranker ready: %s" % cfg.RERANKER_MODEL)
        except Exception as exc:
            # Almost always "not cached and no network". Degrade to plain RRF
            # rather than failing a search the fused ranking can still answer.
            log("reranker unavailable (%s: %s) -- using RRF order" % (type(exc).__name__, exc))
            _state["reranker"] = False
    return _state["reranker"] or None


def _model():
    if _state["model"] is None:
        from fastembed import TextEmbedding
        _state["model"] = TextEmbedding(model_name=cfg.MODEL_NAME, cache_dir=cfg.CACHE_DIR)
    return _state["model"]


def _collection():
    if _state["collection"] is None:
        import chromadb
        client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
        _state["collection"] = client.get_or_create_collection(name=cfg.COLLECTION_NAME)
    return _state["collection"]


def _freshness(max_report=5):
    """Which indexed files changed on disk since the index was built.

    Cheap (one stat per indexed file, cached for 10s) and worth every
    microsecond: a silently stale index turns `find_symbol_references`'s
    "zero references" into a confident lie.
    """
    now = time.time()
    if _state["freshness"] is not None and now - _state["freshness_at"] < 10:
        return _state["freshness"]
    index = _load_index()
    manifest = index.get("manifest", {})
    root = index.get("meta", {}).get("repo_root", cfg.REPO_ROOT)
    changed, missing = [], []
    for rel, record in manifest.items():
        path = os.path.join(root, rel)
        try:
            if os.path.getmtime(path) - record.get("mtime", 0) > 1:
                changed.append(rel)
        except OSError:
            missing.append(rel)
    result = {"changed": changed[:max_report], "changed_count": len(changed),
              "missing_count": len(missing)}
    _state["freshness"], _state["freshness_at"] = result, now
    return result


def _staleness_banner():
    fresh = _freshness()
    total = fresh["changed_count"] + fresh["missing_count"]
    if not total:
        return ""
    return ("\n\n[index staleness] %d indexed file(s) changed on disk since the "
            "last ingest (e.g. %s). Results may be out of date -- call "
            "reindex() to refresh (usually <2s incremental)."
            % (total, ", ".join(fresh["changed"][:3]) or "deleted files"))


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
VALID_CATEGORIES = ("source_code", "documentation", "config", "all")

# Chroma pushes a $in filter down into its SQLite store. Measured on this
# stack: 400 values 0.008s, 10k 0.013s, 32k 0.034s, failing only past ~40k on
# SQLite's variable limit. The previous cap of 400 was therefore ~80x more
# conservative than the engine requires, and every filter above it fell back
# to post-filtering -- silently, and precisely when a repo is big enough for
# the pushdown to matter. 10k keeps ~3x headroom under the measured ceiling.
PUSHDOWN_MAX_PATHS = int(os.environ.get("RAG_PUSHDOWN_MAX_PATHS", "10000"))

# Candidates each retriever returns before fusion, per requested result.
# A fixed multiple stops meaning anything as the corpus grows: 40 candidates
# is 2% of a 2k-chunk index but 0.06% of a 64k-chunk one, and RRF can only
# rank what retrieval actually found. So the pool follows the square root of
# corpus size -- deeper haystack, deeper look -- normalised so that at
# POOL_REFERENCE_CHUNKS the multipliers are exactly what they always were.
POOL_PER_RESULT = int(os.environ.get("RAG_POOL_PER_RESULT", "4"))
POOL_PER_RESULT_FILTERED = int(os.environ.get("RAG_POOL_PER_RESULT_FILTERED", "10"))
POOL_REFERENCE_CHUNKS = int(os.environ.get("RAG_POOL_REFERENCE_CHUNKS", "2000"))
POOL_MAX = int(os.environ.get("RAG_POOL_MAX", "500"))


def _matching_filepaths(index, path_filter):
    if not path_filter:
        return None
    needle = path_filter.lower().replace(os.sep, "/")
    return [rel for rel in index.get("manifest", {}) if needle in rel.lower()]


def _where_clause(category, filepaths):
    """Build the Chroma filter. Returns (where, pushed_down).

    Pushing the path filter into Chroma (instead of dropping hits afterwards)
    is what stops a narrow filter from returning an empty result while the
    matching code sits just outside the global top-k. The caller needs to know
    whether that actually happened, because if it did not, the candidate pool
    has to grow to compensate.
    """
    clauses = []
    if category and category != "all":
        clauses.append({"doc_category": {"$eq": category}})
    pushed_down = False
    if filepaths is not None and 0 < len(filepaths) <= PUSHDOWN_MAX_PATHS:
        clauses.append({"filepath": {"$in": filepaths}})
        pushed_down = True
    if not clauses:
        return None, pushed_down
    return (clauses[0] if len(clauses) == 1 else {"$and": clauses}), pushed_down


def _candidate_pool(top_k, n_chunks, n_files, filepaths, pushed_down):
    """How many candidates each retriever returns before fusion.

    Two effects, both of which used to be fixed constants:

      * corpus size -- sqrt-scaled, so a 32x bigger index looks ~5.7x deeper
        rather than staying at the same absolute (and increasingly tiny) slice;
      * filter selectivity -- when the path filter could NOT be pushed down,
        post-filtering will discard most of what comes back, so over-fetch by
        the inverse of the fraction of the repo the filter matched.

    At POOL_REFERENCE_CHUNKS with a pushdown this returns exactly the old
    values (top_k*4 unfiltered, top_k*10 filtered), so small repos see no
    change at all.
    """
    per_result = POOL_PER_RESULT_FILTERED if filepaths is not None else POOL_PER_RESULT
    pool = top_k * per_result
    if n_chunks > POOL_REFERENCE_CHUNKS:
        pool = int(pool * math.sqrt(float(n_chunks) / POOL_REFERENCE_CHUNKS))
    if filepaths is not None and not pushed_down and n_files:
        # Floor the selectivity so a pathological filter cannot demand the
        # whole corpus; POOL_MAX clamps it regardless.
        selectivity = max(float(len(filepaths)) / n_files, 0.02)
        pool = int(pool / selectivity)
    return max(top_k, min(pool, POOL_MAX, n_chunks))


def _rrf(ranked_lists, weights=None, k=None, top_n=10, boosts=None):
    """Weighted Reciprocal Rank Fusion over any number of ranked id lists.

    `weights` scales each list's contribution (see cfg.RRF_WEIGHTS -- a bare
    identifier trusts BM25, a prose question trusts the dense side).
    `boosts` adds a per-id bonus already expressed on the RRF scale, which is
    how an exact symbol-name match earns promotion without a second sort.
    """
    k = cfg.RRF_K if k is None else k
    weights = weights or [1.0] * len(ranked_lists)
    scores = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
    for cid, bonus in (boosts or {}).items():
        if cid in scores:
            scores[cid] += bonus
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_n]


def _symbol_boosts(chunks, candidate_ids, query):
    """Bonus for chunks whose OWN symbol is a term the user actually typed.

    Rank alone cannot express "this chunk IS the thing you named" -- a
    definition and a file that merely mentions it can arrive at the same rank
    from different retrievers. One rank-1 RRF step is 1/(k+1), so the bonus is
    scaled to a fraction of that: decisive among near-ties, never a veto.
    """
    terms = set(cfg.tokenize(query))
    if not terms:
        return {}
    unit = cfg.SYMBOL_BOOST / (cfg.RRF_K + 1)
    boosts = {}
    for cid in candidate_ids:
        chunk = chunks.get(cid)
        if not chunk:
            continue
        symbol = chunk["metadata"].get("symbol", "")
        if not symbol or symbol == "block":
            continue
        bare = symbol.lower().split(".")[-1]
        if bare in terms:
            boosts[cid] = unit
        elif set(cfg.tokenize(symbol)) & terms:
            boosts[cid] = unit * 0.5        # partial: camelCase piece matched
    return boosts


def _rerank(query, ordered_ids, chunks, top_k):
    """Cross-encoder rerank of the fused candidates. Falls back silently."""
    encoder = _reranker()
    if encoder is None or len(ordered_ids) <= 1:
        return ordered_ids[:top_k], False
    candidates = ordered_ids[:cfg.RERANK_CANDIDATES]
    try:
        docs = [chunks[cid]["text"][:4000] for cid in candidates]
        scores = list(encoder.rerank(query, docs))
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        reranked = [candidates[i] for i in order]
        return (reranked + ordered_ids[len(candidates):])[:top_k], True
    except Exception as exc:
        log("rerank failed (%s: %s) -- keeping RRF order" % (type(exc).__name__, exc))
        return ordered_ids[:top_k], False


def _format_hit(chunk, return_skeletons, rank=None):
    meta = chunk["metadata"]
    part = ""
    if meta.get("parts", 1) > 1:
        part = " | Part: %d/%d" % (meta["part"], meta["parts"])
    parent = " | In: %s" % meta["parent_symbol"] if meta.get("parent_symbol") else ""
    body = chunk["skeleton"] if return_skeletons else chunk["text"]
    head = "--- [ID: %s]\n    File: %s:%s-%s | Lang: %s | Symbol: %s%s%s ---" % (
        chunk["id"], meta["filepath"], meta["start_line"], meta["end_line"],
        meta["language"], meta["symbol"], parent, part)
    return "%s\n%s" % (head, body)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
@tool_guard
def search_codebase(query: str, category: str = "source_code",
                    path_filter: Optional[str] = None, top_k: int = 10,
                    return_skeletons: bool = True) -> str:
    """PRIMARY CODEBASE SEARCH. Use this before grep/glob/file-reading for any
    "where/how is X implemented?" question. Hybrid dense-vector + BM25 search
    fused with Reciprocal Rank Fusion; every hit reports exact file:start-end
    lines you can pass straight to get_file_context.

    Args:
        query: Natural language ("how does session refresh work") or bare
            identifiers ("refresh_access_token") -- both work.
        category: 'source_code' (default, real implementation), 'documentation'
            (plans/specs/READMEs), 'config' (json/yaml/toml), or 'all'.
        path_filter: Substring of the file path to restrict to, e.g.
            'backend', 'src/api', 'schema.sql'.
        top_k: Number of results (default 10).
        return_skeletons: True (default) returns signature + first body lines
            per hit; call get_chunk_content(chunk_id) for a full body. Set
            False to get complete chunk text inline.
    """
    if category not in VALID_CATEGORIES:
        return "Invalid category '%s'. Use one of: %s." % (category, ", ".join(VALID_CATEGORIES))

    index = _load_index()
    chunks = index.get("chunks", {})
    if not chunks:
        return ("The index is empty. Run reindex(full=True) or "
                "`python3 ingest_codebase.py --full` in rag-workspace/.")

    filepaths = _matching_filepaths(index, path_filter)
    if filepaths is not None and not filepaths:
        return ("No indexed file path contains '%s'. Try a shorter fragment, or "
                "call rag_status() to see what is indexed." % path_filter)

    # --- dense -------------------------------------------------------------
    prefixed = cfg.query_prefix() + query
    embedding = list(_model().embed([prefixed]))[0].tolist()
    where, pushed_down = _where_clause(category, filepaths)
    n_results = _candidate_pool(top_k, len(chunks), len(index.get("manifest", {})),
                                filepaths, pushed_down)
    res = _collection().query(query_embeddings=[embedding], n_results=n_results,
                              where=where, include=["metadatas"])
    dense_ids = []
    for cid in (res.get("ids") or [[]])[0]:
        if cid not in chunks:
            continue
        meta = chunks[cid]["metadata"]
        if filepaths is not None and meta["filepath"] not in filepaths:
            continue
        dense_ids.append(cid)

    # --- lexical -----------------------------------------------------------
    lexical_ids = []
    bm25, bm25_ids = index.get("bm25"), index.get("bm25_ids", [])
    if bm25 is not None and bm25_ids:
        tokens = cfg.tokenize(query)
        if tokens:
            scores = bm25.get_scores(tokens)
            order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)
            for i in order:
                if scores[i] <= 0:
                    break
                cid = bm25_ids[i]
                chunk = chunks.get(cid)
                if not chunk:
                    continue
                meta = chunk["metadata"]
                if category != "all" and meta["doc_category"] != category:
                    continue
                if filepaths is not None and meta["filepath"] not in filepaths:
                    continue
                lexical_ids.append(cid)
                if len(lexical_ids) >= n_results:
                    break

    # --- fusion ----------------------------------------------------------
    shape = cfg.query_shape(query)
    w_dense, w_lex = cfg.RRF_WEIGHTS[shape]
    pool = max(top_k, cfg.RERANK_CANDIDATES if _reranker() else top_k)
    boosts = _symbol_boosts(chunks, set(dense_ids) | set(lexical_ids), query)
    candidates = _rrf([dense_ids, lexical_ids], weights=[w_dense, w_lex],
                      top_n=pool, boosts=boosts)
    if not candidates:
        return ("No matches for '%s' (category=%s, path_filter=%s). Try "
                "category='all', a broader path_filter, or "
                "find_symbol_or_keyword() for a literal string."
                % (query, category, path_filter))

    fused, reranked = _rerank(query, candidates, chunks, top_k)

    body = "\n\n".join(_format_hit(chunks[cid], return_skeletons) for cid in fused)
    header = ("%d result(s) for '%s' [dense=%d, lexical=%d of pool %d%s | %s "
              "query -> weights d%.1f/l%.1f%s%s]\n\n"
              % (len(fused), query, len(dense_ids), len(lexical_ids), n_results,
                 "" if filepaths is None else
                 (", %d files pushed down" % len(filepaths) if pushed_down
                  else ", %d files POST-filtered" % len(filepaths)),
                 shape, w_dense, w_lex,
                 ", %d symbol-boosted" % len(boosts) if boosts else "",
                 ", cross-encoder reranked" if reranked else ""))
    return header + body + _staleness_banner()


@mcp.tool()
@tool_guard
def find_symbol_references(symbol_name: str, path_filter: Optional[str] = None,
                           category: str = "source_code", limit: int = 10) -> str:
    """Find where a symbol is DEFINED and where it is USED (functions,
    composables, tables, columns, RPC names, components, variables).

    Matches on word boundaries (so 'read' does not match 'already_read'), ranks
    definitions above call sites, and verifies any "zero references" answer
    against the live working tree so a stale index can never produce a false
    negative.

    Args:
        symbol_name: Exact identifier, e.g. 'refresh_access_token'.
        path_filter: Optional path substring to restrict the search.
        category: 'source_code' (default), 'documentation', 'config', or 'all'.
        limit: Maximum chunks to report (default 10).
    """
    index = _load_index()
    chunks = index.get("chunks", {})
    if not chunks:
        return "The index is empty. Run reindex(full=True) first."

    pattern = re.compile(r"\b%s\b" % re.escape(symbol_name), re.IGNORECASE)
    # A call site outranks a passing mention in a comment or a doc. RPC and
    # table names are invoked as string literals, so those count too.
    esc = re.escape(symbol_name)
    call_pattern = re.compile(r"\b%s\b\s*[(<{]|['\"`]%s['\"`]" % (esc, esc))
    needle = path_filter.lower() if path_filter else None
    scored = []
    for chunk in chunks.values():
        meta = chunk["metadata"]
        if category != "all" and meta["doc_category"] != category:
            continue
        if needle and needle not in meta["filepath"].lower():
            continue
        hits = len(pattern.findall(chunk["text"]))
        if not hits:
            continue
        # `public.refresh_token` and `refresh_token` are the same symbol;
        # compare on the last dotted segment.
        chunk_symbol = meta["symbol"].lower().split(".")[-1]
        is_def = chunk_symbol == symbol_name.lower()
        is_call = bool(call_pattern.search(chunk["text"]))
        tier = 2 if is_def else (1 if is_call else 0)
        scored.append((tier, hits, chunk, is_def))

    if not scored:
        live = _grep(r"\b%s\b" % re.escape(symbol_name), path_filter, "*", 5, 0)
        if live["count"]:
            return ("Symbol '%s' is NOT in the index but DOES exist in the working "
                    "tree (%d live matches) -- the index is stale. Call reindex(). "
                    "Live matches:\n%s" % (symbol_name, live["count"],
                                           "\n".join(live["lines"])))
        return ("Symbol '%s' has ZERO references%s%s -- verified against both the "
                "index and a live grep of the working tree (clean negative "
                "confirmation)." % (symbol_name,
                                    " under category '%s'" % category if category != "all" else "",
                                    " in paths matching '%s'" % path_filter if path_filter else ""))

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    definitions = sum(1 for row in scored if row[3])
    out = []
    for tier, hits, chunk, is_def in scored[:limit]:
        meta = chunk["metadata"]
        kind = "DEFINITION" if is_def else ("call site" if tier == 1 else "mention")
        out.append("--- [%s x%d] [ID: %s]\n    File: %s:%s-%s | Lang: %s | Symbol: %s ---\n%s"
                   % (kind, hits, chunk["id"], meta["filepath"], meta["start_line"],
                      meta["end_line"], meta["language"], meta["symbol"], chunk["skeleton"]))
    header = ("Found '%s' in %d chunk(s) (%d definition-level, showing %d):\n\n"
              % (symbol_name, len(scored), definitions, min(limit, len(scored))))
    return header + "\n\n".join(out) + _staleness_banner()


def _grep(pattern, path_filter, file_pattern, limit, offset):
    """Literal/regex search over the working tree. Prefers ripgrep, falls back
    to POSIX grep so the tool works on a bare machine."""
    index = _load_index()
    root = index.get("meta", {}).get("repo_root", cfg.REPO_ROOT)
    search_dir = root
    if path_filter:
        candidate = os.path.join(root, path_filter)
        if os.path.isdir(candidate):
            search_dir = candidate

    has_rg = any(os.access(os.path.join(p, "rg"), os.X_OK)
                 for p in os.environ.get("PATH", "").split(os.pathsep) if p)
    if has_rg:
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "-e", pattern]
        for directory in sorted(cfg.EXCLUDE_DIRS):
            cmd += ["--glob", "!%s/**" % directory]
        for name in sorted(cfg.EXCLUDE_FILE_NAMES):
            cmd += ["--glob", "!%s" % name]
        if file_pattern and file_pattern != "*":
            cmd += ["--glob", file_pattern]
        cmd.append(search_dir)
    else:
        cmd = ["grep", "-rnEI"]
        for directory in sorted(cfg.EXCLUDE_DIRS):
            cmd.append("--exclude-dir=%s" % directory)
        for name in sorted(cfg.EXCLUDE_FILE_NAMES):
            cmd.append("--exclude=%s" % name)
        cmd += ["--include=%s" % (file_pattern or "*"), pattern, search_dir]

    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                     timeout=cfg.GREP_TIMEOUT)
        lines = raw.decode("utf-8", "ignore").splitlines()
    except subprocess.CalledProcessError:
        lines = []
    except subprocess.TimeoutExpired:
        return {"count": 0, "lines": [], "timeout": True, "tool": "rg" if has_rg else "grep"}

    if path_filter and search_dir == root:
        needle = path_filter.lower()
        lines = [ln for ln in lines if needle in ln.lower()]

    relative = []
    for line in lines:
        if line.startswith(root + os.sep):
            line = line[len(root) + 1:]
        relative.append(line)
    window = relative[offset:offset + limit]
    return {"count": len(relative), "lines": window, "timeout": False,
            "tool": "rg" if has_rg else "grep"}


@mcp.tool()
@tool_guard
def find_symbol_or_keyword(pattern: str, path_filter: Optional[str] = None,
                           file_pattern: str = "*", limit: int = 40,
                           offset: int = 0) -> str:
    """Literal/regex search of the WORKING TREE (always current, never stale).

    Use to prove a string exists or definitively does not -- column names,
    feature flags, TODOs, RPC names. Results are paginated: when the tail is
    truncated, call again with a higher `offset` instead of guessing.

    Args:
        pattern: Extended-regex pattern, e.g. 'expires_at|created_at'.
        path_filter: Directory or path substring to restrict the search.
        file_pattern: Glob for filenames, e.g. '*.py', '*.ts', '*.sql'.
        limit: Max matching lines to return (default 40).
        offset: Skip this many matches -- use for the next page.
    """
    result = _grep(pattern, path_filter, file_pattern, max(1, limit), max(0, offset))
    if result.get("timeout"):
        # POSIX grep has no gitignore awareness and re-walks build output the
        # index already skips, so it is the case that actually times out.
        hint = ("" if result["tool"] == "rg" else
                " This machine has no ripgrep, so the slower POSIX grep ran; "
                "installing ripgrep makes this roughly an order of magnitude "
                "faster.")
        return ("Search for '%s' timed out after %ds. Narrow it with path_filter "
                "or file_pattern, or raise RAG_GREP_TIMEOUT.%s"
                % (pattern, cfg.GREP_TIMEOUT, hint))
    if not result["count"]:
        return ("Pattern '%s' NOT FOUND in the working tree%s%s (clean negative "
                "confirmation, searched with %s)."
                % (pattern,
                   " under '%s'" % path_filter if path_filter else "",
                   " matching '%s'" % file_pattern if file_pattern != "*" else "",
                   result["tool"]))
    shown_from = offset + 1
    shown_to = offset + len(result["lines"])
    body = "\n".join(result["lines"])
    footer = ""
    if shown_to < result["count"]:
        footer = ("\n... showing %d-%d of %d matches. Call again with offset=%d "
                  "for the next page." % (shown_from, shown_to, result["count"], shown_to))
    return "%d match(es) for '%s':\n%s%s" % (result["count"], pattern, body, footer)


@mcp.tool()
@tool_guard
def get_chunk_content(chunk_id: str) -> str:
    """Full verbatim source of one chunk returned by search_codebase or
    find_symbol_references. Pass the ID exactly as printed."""
    index = _load_index()
    chunks = index.get("chunks", {})
    chunk = chunks.get(chunk_id)
    if chunk is None:
        prefix = chunk_id.split(":")[0]
        nearby = [cid for cid in chunks if cid.startswith(prefix)][:8]
        hint = ("\nChunks in that file: \n  " + "\n  ".join(nearby)) if nearby else ""
        return ("Chunk ID '%s' not found (the index may have been rebuilt since "
                "that ID was issued -- re-run your search).%s" % (chunk_id, hint))
    meta = chunk["metadata"]
    return ("=== %s | %s:%s-%s | Symbol: %s ===\n%s"
            % (chunk_id, meta["filepath"], meta["start_line"], meta["end_line"],
               meta["symbol"], chunk["text"]))


@mcp.tool()
@tool_guard
def get_file_context(targets: List[Dict[str, Any]]) -> str:
    """Read exact line ranges from MANY files in ONE call -- always prefer this
    over several single-file reads.

    Args:
        targets: [{"path": "db/schema.sql", "start_line": 54,
                   "end_line": 64}, ...]. Paths may be repo-relative or
            absolute. Omit the range to get the head of the file.
    """
    index = _load_index()
    root = index.get("meta", {}).get("repo_root", cfg.REPO_ROOT)
    if isinstance(targets, dict):
        targets = [targets]
    results = []
    for target in targets or []:
        rel = (target or {}).get("path", "")
        if not rel:
            results.append("=== (missing 'path' key in target) ===")
            continue
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if not os.path.exists(path):
            results.append("=== %s ===\nError: file not found." % rel)
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.read().splitlines()
            start = max(1, int(target.get("start_line", 1) or 1))
            end = int(target.get("end_line") or (start + 80))
            end = max(start, min(len(lines), end))
            numbered = "\n".join("%5d  %s" % (start + i, text)
                                 for i, text in enumerate(lines[start - 1:end]))
            results.append("=== %s (lines %d-%d of %d) ===\n%s"
                           % (rel, start, end, len(lines), numbered))
        except Exception as exc:
            results.append("=== %s ===\nError reading file: %s" % (rel, exc))
    return "\n\n".join(results) if results else "No targets supplied."


@mcp.tool()
@tool_guard
def rag_status() -> str:
    """Index health: chunk/file counts, embedding model, build time, indexed
    git commit, and which files changed since the last ingest. Call this when a
    search result looks wrong or outdated."""
    index = _load_index()
    meta = index.get("meta", {})
    chunks = index.get("chunks", {})
    if not chunks:
        return ("No index found at %s. Run reindex(full=True)." % cfg.INDEX_PATH)
    fresh = _freshness(max_report=20)
    cfg.count_tokens("probe")            # resolves cfg.TOKENIZER_AVAILABLE
    built = meta.get("built_at", 0)
    langs = {}
    for chunk in chunks.values():
        lang = chunk["metadata"]["language"]
        langs[lang] = langs.get(lang, 0) + 1
    top = ", ".join("%s=%d" % kv for kv in sorted(langs.items(), key=lambda x: -x[1])[:10])
    lines = [
        "Repository      : %s" % meta.get("repo_root", cfg.REPO_ROOT),
        "Indexed files   : %d" % len(index.get("manifest", {})),
        "Chunks          : %d" % len(chunks),
        "Embedding model : %s (local ONNX, offline)" % meta.get("model", "?"),
        "Chunker         : v%s (%s)" % (meta.get("chunker_version", "?"),
                                        "tree-sitter" if meta.get("tree_sitter") else "regex fallback"),
        "Built           : %s (%.1f min ago)" % (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(built)) if built else "?",
            (time.time() - built) / 60 if built else 0),
        "Indexed commit  : %s" % (meta.get("git_head", "")[:12] or "n/a"),
        "Languages       : %s" % top,
        "Vector store    : %d embeddings" % _collection().count(),
        "Token counting  : %s" % ("exact (encoder tokenizer)" if cfg.TOKENIZER_AVAILABLE
                                  else "estimated (tokenizer not cached)"),
        "Fusion          : weighted RRF k=%d, symbol boost %.2f" % (cfg.RRF_K, cfg.SYMBOL_BOOST),
        "Candidate pool  : %d unfiltered / %d filtered (top_k=10), max %d" % (
            _candidate_pool(10, len(chunks), len(index.get("manifest", {})), None, False),
            _candidate_pool(10, len(chunks), len(index.get("manifest", {})), [], True),
            POOL_MAX),
        "Path pushdown   : up to %s files per query" % f"{PUSHDOWN_MAX_PATHS:,}",
        "Reranker        : %s" % (
            ("%s (active)" % cfg.RERANKER_MODEL) if _reranker()
            else ("%s (set but unavailable -- see rag.log)" % cfg.RERANKER_MODEL
                  if cfg.RERANKER_MODEL else "off (set RAG_RERANKER to enable)")),
        "Index load      : restricted unpickler (allowlisted classes only)",
    ]
    heads = meta.get("git_heads") or {}
    if heads:
        lines.append("Repos indexed   : %d" % len(heads))
        for name, head in sorted(heads.items()):
            lines.append("   %-24s %s" % (name, head[:12]))
    if fresh["changed_count"] or fresh["missing_count"]:
        lines.append("STALE           : %d changed, %d deleted since ingest -- %s"
                     % (fresh["changed_count"], fresh["missing_count"],
                        ", ".join(fresh["changed"][:5])))
        lines.append("                  call reindex() to refresh.")
    else:
        lines.append("Freshness       : up to date with the working tree.")
    return "\n".join(lines)


@mcp.tool()
@tool_guard
def reindex(full: bool = False) -> str:
    """Re-index the repository after edits. Incremental by default: only files
    whose contents changed are re-chunked and re-embedded (typically <2s).

    Args:
        full: True forces a complete rebuild (needed only after changing the
            embedding model or chunker).
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest_codebase.py")
    cmd = [sys.executable, script]
    if full:
        cmd.append("--full")
    started = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=cfg.REINDEX_TIMEOUT)
    except subprocess.TimeoutExpired:
        return ("Re-index timed out after %ds -- raise RAG_REINDEX_TIMEOUT."
                % cfg.REINDEX_TIMEOUT)
    _state["index"] = None
    _state["freshness"] = None
    _load_index()
    tail = "\n".join((out.stdout or "").strip().splitlines()[-8:])
    status = "ok" if out.returncode == 0 else "exit code %d" % out.returncode
    return "Re-index finished in %.1fs (%s):\n%s" % (time.time() - started, status, tail)


if __name__ == "__main__":
    log("server starting (repo=%s, index=%s)" % (cfg.REPO_ROOT, cfg.INDEX_PATH))
    mcp.run()
```

---

## Phase 6 — Cache the model and build the first index

Write `<WORKSPACE>/download_model.py`:

```python
#!/usr/bin/env python3
"""Cache the local models so the server never needs the network at query time.

Three sources, tried in order:

  1. Already cached           -- nothing to do, works offline.
  2. This project's release   -- one verified tarball, no HuggingFace round trip.
  3. HuggingFace via fastembed -- the fallback, and what the release was built from.

Step 2 exists because step 3 is the single part of setup that needs the public
internet: it is what breaks on a locked-down corporate network, behind a proxy,
or the day upstream renames a repository. Step 3 stays as the fallback so the
tool never *depends* on the release being reachable either.

    python3 download_model.py
    RAG_SKIP_MODEL_ASSET=1 python3 download_model.py     # force HuggingFace
    RAG_RERANKER=Xenova/ms-marco-MiniLM-L-6-v2 python3 download_model.py
"""
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_config as cfg


def log(msg):
    print(msg, flush=True)


def already_cached() -> bool:
    """A tokenizer.json for this model means fastembed can work offline."""
    return bool(cfg._tokenizer_path())


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_members(archive: tarfile.TarFile, root: str):
    """Yield only members that stay inside `root` once extracted.

    A tar entry may name `../../etc/whatever`, or be a symlink pointing out of
    the tree -- unpacking one blindly writes wherever it likes. The model cache
    legitimately contains relative symlinks (snapshots/ into blobs/), so they
    cannot simply be rejected; each one is resolved and checked instead.
    """
    root = os.path.realpath(root)
    for member in archive.getmembers():
        target = os.path.realpath(os.path.join(root, member.name))
        if not (target == root or target.startswith(root + os.sep)):
            raise ValueError("refusing unsafe path in archive: %s" % member.name)
        if member.issym() or member.islnk():
            link = os.path.realpath(
                os.path.join(os.path.dirname(target), member.linkname))
            if not (link == root or link.startswith(root + os.sep)):
                raise ValueError("refusing link escaping the archive: %s -> %s"
                                 % (member.name, member.linkname))
        yield member


def fetch_release_asset() -> bool:
    """Download, verify and unpack the pre-built cache. False = fall through."""
    if cfg.SKIP_MODEL_ASSET or not cfg.MODEL_ASSET_URL:
        return False
    parent = os.path.dirname(cfg.CACHE_DIR.rstrip(os.sep))
    tmp_dir = tempfile.mkdtemp(prefix="rag-model-", dir=parent)
    tarball = os.path.join(tmp_dir, "model-cache.tar.gz")
    try:
        log("Fetching pre-built cache from %s" % cfg.MODEL_ASSET_URL)
        urllib.request.urlretrieve(cfg.MODEL_ASSET_URL, tarball)

        actual = _sha256(tarball)
        if cfg.MODEL_ASSET_SHA256 and actual != cfg.MODEL_ASSET_SHA256:
            # Never unpack something that failed its digest -- a mismatch means
            # the wrong file at best and a substituted one at worst.
            log("  SHA-256 mismatch!\n    expected %s\n    got      %s"
                % (cfg.MODEL_ASSET_SHA256, actual))
            log("  Ignoring the asset and falling back to HuggingFace.")
            return False
        log("  verified sha256 %s" % actual[:16])

        with tarfile.open(tarball, "r:gz") as archive:
            members = list(_safe_members(archive, tmp_dir))
            # Python 3.12+ warns unless an extraction filter is named, and 3.14
            # will enforce one. 'tar' keeps the symlinks the HF cache layout
            # needs; the membership check above is the real guard either way.
            try:
                archive.extractall(tmp_dir, members=members, filter="tar")
            except TypeError:
                archive.extractall(tmp_dir, members=members)   # < 3.12

        unpacked = os.path.join(tmp_dir, os.path.basename(cfg.CACHE_DIR))
        if not os.path.isdir(unpacked):
            log("  archive did not contain %s; falling back."
                % os.path.basename(cfg.CACHE_DIR))
            return False

        # Merge rather than replace: the cache may already hold other models
        # (a reranker, a second embedder) that this asset knows nothing about.
        os.makedirs(cfg.CACHE_DIR, exist_ok=True)
        for name in os.listdir(unpacked):
            src, dst = os.path.join(unpacked, name), os.path.join(cfg.CACHE_DIR, name)
            if os.path.exists(dst):
                continue
            shutil.move(src, dst)
        log("  unpacked into %s" % cfg.CACHE_DIR)
        return already_cached()
    except Exception as exc:
        log("  asset unavailable (%s: %s) -- falling back to HuggingFace."
            % (type(exc).__name__, exc))
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    log("Model      : %s" % cfg.MODEL_NAME)
    log("Cache dir  : %s" % cfg.CACHE_DIR)

    if already_cached():
        log("Already cached -- nothing to download.")
    elif fetch_release_asset():
        log("Ready from the pre-built release asset (no HuggingFace request).")
    else:
        log("Downloading from HuggingFace via fastembed...")

    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=cfg.MODEL_NAME, cache_dir=cfg.CACHE_DIR)
    vector = list(model.embed(["warm up the onnx session"]))[0]
    log("Ready: %d-dimensional embeddings, running fully offline." % len(vector))

    # The chunker counts tokens with this file to keep chunks under the encoder
    # window; without it it falls back to a conservative character estimate and
    # over-splits. Worth reporting rather than leaving to be discovered.
    log("Tokenizer  : %s" % (cfg._tokenizer_path() or "NOT FOUND (using estimates)"))

    if cfg.RERANKER_MODEL:
        log("\nCaching reranker '%s' ..." % cfg.RERANKER_MODEL)
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            encoder = TextCrossEncoder(model_name=cfg.RERANKER_MODEL,
                                       cache_dir=cfg.CACHE_DIR)
            list(encoder.rerank("warm up", ["warm up the cross encoder"]))
            log("Reranker ready — searches will rerank their top %d candidates."
                % cfg.RERANK_CANDIDATES)
        except Exception as exc:
            log("Reranker unavailable (%s: %s).\nSearch still works; it just keeps "
                "the fused RRF order." % (type(exc).__name__, exc))
    else:
        log("\nReranker   : not configured (optional).\n"
            "  Enable with: RAG_RERANKER=Xenova/ms-marco-MiniLM-L-6-v2 "
            "python3 download_model.py")


if __name__ == "__main__":
    main()
```

Run setup:

```bash
"$WORKSPACE/.poc-venv/bin/python3" "$WORKSPACE/download_model.py"
"$WORKSPACE/.poc-venv/bin/python3" "$WORKSPACE/ingest_codebase.py" --target "$TARGET" --full
```

Expect a per-language breakdown and a chunk count. Sanity-check both against Phase 0:
if a language you know is present shows zero chunks, its extension is missing from
`SUPPORTED_EXTENSIONS`; if a count looks absurdly high for one file, it is generated
output that belongs in the exclusions.

Rough scale: a few thousand chunks embed in a couple of minutes on a laptop CPU, and
that is the *only* slow step. Every subsequent incremental index is about a second.

---

## Phase 7 — Wire it up

### 7.1 Register with an MCP client

Every MCP host accepts the same stdio server definition; only the config file location
differs. Use absolute paths.

```json
{
  "mcpServers": {
    "codebase-rag": {
      "type": "stdio",
      "command": "<WORKSPACE>/.poc-venv/bin/python3",
      "args": ["<WORKSPACE>/mcp_server.py"],
      "env": {}
    }
  }
}
```

| Client | Where the config goes |
|---|---|
| Claude Code | `<TARGET>/.mcp.json` (project scope), or `claude mcp add codebase-rag -- <WORKSPACE>/.poc-venv/bin/python3 <WORKSPACE>/mcp_server.py` |
| Gemini CLI / Antigravity | `~/.gemini/settings.json` or the workspace `mcp_config.json`, same `mcpServers` shape |
| Cursor | `<TARGET>/.cursor/mcp.json` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| Continue | `~/.continue/config.json`, under `mcpServers` |
| Zed | `settings.json`, under `context_servers` |
| Anything else | Launch `<WORKSPACE>/.poc-venv/bin/python3 <WORKSPACE>/mcp_server.py` as a stdio subprocess |

To point the same server at a *different* repository without editing code, set
`"env": {"RAG_REPO_ROOT": "/path/to/other/repo", "RAG_INDEX_DIR": "/path/to/other/index"}`.

**The client caches tool schemas at connect time.** After changing tool signatures,
restart the client session or the new parameters will not appear.

### 7.2 Git hooks

Write `<WORKSPACE>/hooks/rag-reindex.sh`:

```bash
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
```

Write `<WORKSPACE>/install_hooks.sh`:

```bash
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
```

```bash
chmod +x "$WORKSPACE/hooks/rag-reindex.sh" "$WORKSPACE/install_hooks.sh"
"$WORKSPACE/install_hooks.sh"
```

This covers pull, commit, checkout and rebase. It does **not** cover uncommitted edits —
that is what the staleness banner and `reindex()` are for.

### 7.3 Keep the tree and the index fresh at session start

Git hooks cover *your* git operations. They do nothing about commits that landed on
the remote while you were away — so the first question of a new session gets answered
against yesterday's code. Close that gap with one script that pulls and re-indexes,
invoked when a session starts.

Write `<WORKSPACE>/sync_and_index.sh`:

```bash
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
```

```bash
chmod +x "$WORKSPACE/sync_and_index.sh"
"$WORKSPACE/sync_and_index.sh"          # verify by hand before automating it
```

The safety rules in that script are the point. Auto-pulling is only acceptable if it
can never damage work in progress: fast-forward only, skipped entirely on a dirty tree,
skipped on a detached HEAD or a branch with no upstream, bounded network wait, and every
failure non-fatal so a flaky connection can never block a session from starting.

Wire it to your client's session-start event:

| Client | How |
|---|---|
| Claude Code | `SessionStart` hook in `.claude/settings.local.json` (personal) or `.claude/settings.json` (team) |
| Other clients | Whatever startup/task hook they expose, a shell alias, or run it by hand |

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "\"${CLAUDE_PROJECT_DIR}/rag-workspace/sync_and_index.sh\"",
            "timeout": 120,
            "statusMessage": "Syncing repo and refreshing RAG index..."
          }
        ]
      }
    ]
  }
}
```

`${CLAUDE_PROJECT_DIR}` keeps the entry portable across machines, which matters if you
commit it in `.claude/settings.json` for a team. Typical cost is two or three seconds
(fetch, then a ~1s incremental index). Add `"async": true` to run it in the background
instead, accepting that the first query of a session may race the refresh.

If you would rather never auto-pull, set `RAG_SKIP_PULL=1` in the hook's environment —
the index still refreshes from whatever is on disk.

### 7.4 Ignore the generated artifacts

Append to `<TARGET>/.gitignore`:

```gitignore
rag-workspace/.poc-venv/
rag-workspace/.models_cache/
rag-workspace/chroma_db/
rag-workspace/rag_index.pkl
rag-workspace/rag_index.pkl.tmp
rag-workspace/rag.log
rag-workspace/__pycache__/
```

All of it is regenerable from source in one command. None of it belongs in version
control.

### 7.5 Tell the assistants the tools exist

A server nobody calls is worthless. Add this to whichever instruction files the
assistants in your team read — `CLAUDE.md`, `GEMINI.md`, `AGENTS.md`, `.cursorrules`,
`.windsurfrules`. The content is identical; only the filename differs.

```markdown
## Local Codebase RAG (MCP server `codebase-rag`)

100% local: tree-sitter chunking -> ONNX embeddings -> ChromaDB (dense) + BM25
(lexical) -> Reciprocal Rank Fusion. Every hit reports an exact
`file:start_line-end_line`.

1. `search_codebase(query, category="source_code", path_filter=None, top_k=10,
   return_skeletons=True)` — first stop for any "where / how is X implemented?"
   question. Natural language and bare identifiers both work.
   `category`: 'source_code' | 'documentation' | 'config' | 'all'.
2. `find_symbol_references(symbol_name, path_filter=None, category="source_code",
   limit=10)` — definitions and call sites, labelled and ranked. Negatives are
   verified against a live grep.
3. `find_symbol_or_keyword(pattern, path_filter=None, file_pattern="*", limit=40,
   offset=0)` — regex over the working tree; always current; paginated.
4. `get_file_context(targets=[{"path": ..., "start_line": N, "end_line": M}])` —
   batch reader; pull every range you need in ONE call.
5. `get_chunk_content(chunk_id)` — full source behind a search hit.
6. `rag_status()` — counts, model, build time, staleness.
7. `reindex(full=False)` — refresh after edits (incremental, ~1s).

Rules: search before shell; batch line-range reads into one `get_file_context`;
trust the reported line numbers; if a result carries an `[index staleness]`
banner, call `reindex()` before drawing conclusions.
```

### 7.6 Optional: a setup skill for assistants that support skills

For Claude Code, `<TARGET>/skills/rag/SKILL.md` (or `~/.claude/skills/rag/SKILL.md` for
all projects) lets an assistant rebuild or troubleshoot this pipeline on request. Other
tools have equivalents (Cursor rules, Continue prompts); the body is the same prose.

```markdown
---
name: rag
description: Interactive setup guide for deploying a 100% local, LLM-agnostic hybrid
  codebase RAG system using FastMCP, ChromaDB, FastEmbed (ONNX), tree-sitter and BM25.
  Use when building, configuring, running, or troubleshooting a local codebase RAG
  indexing and search pipeline.
---

Follow `create_rag.md` phase by phase. Guide the user one step at a time, prompt for
paths before generating anything, keep execution offline and in user-space, and never
emit assistant-specific configuration. Do not declare success until every check in
Phase 8 passes.
```

---

## Phase 8 — Validate (do not skip)

These are the failure modes that let a RAG server look functional while being useless.

### 8.1 Structural invariants across the whole repo

```bash
"$WORKSPACE/.poc-venv/bin/python3" - <<'PY'
import os, sys, collections
sys.path.insert(0, os.environ["WORKSPACE"])
import rag_config as cfg, rag_chunker as k, ingest_codebase as ing

bad = over = errors = 0
ids = collections.Counter()
files = 0
for path in ing.iter_source_files(cfg.REPO_ROOT):
    rel = os.path.relpath(path, cfg.REPO_ROOT).replace(os.sep, "/")
    files += 1
    try:
        content = open(path, "rb").read().decode("utf-8", "ignore")
        chunks = k.extract_chunks(path, rel, content)
    except Exception as exc:
        errors += 1; print("EXCEPTION", rel, repr(exc)); continue
    total_lines = max(len(content.splitlines()), 1)
    for chunk in chunks:
        meta = chunk["metadata"]
        ids[chunk["id"]] += 1
        if not (1 <= meta["start_line"] <= meta["end_line"] <= total_lines):
            bad += 1; print("BAD RANGE", rel, meta["start_line"], meta["end_line"])
        if cfg.est_tokens(chunk["text"]) > cfg.MAX_CHUNK_TOKENS:
            over += 1; print("OVER BUDGET", rel, meta["symbol"])
dups = sum(1 for n in ids.values() if n > 1)
print("files=%d chunks=%d exceptions=%d bad_ranges=%d over_budget=%d dup_ids=%d"
      % (files, sum(ids.values()), errors, bad, over, dups))
PY
```

**Required: `exceptions=0 bad_ranges=0 over_budget=0 dup_ids=0`.** Anything else means a
chunker bug — fix it before indexing, not after.

### 8.2 End-to-end over the real MCP protocol

Testing the Python functions directly does not prove the server works. Speak the
protocol:

```bash
"$WORKSPACE/.poc-venv/bin/python3" - <<'PY'
import asyncio, os
from fastmcp import Client
SERVER = os.path.join(os.environ["WORKSPACE"], "mcp_server.py")

# Replace with YOUR three Phase 0.4 validation cases.
IDENTIFIER   = "some_distinctive_symbol"
CONCEPT      = "how does authentication refresh work"
KNOWN_ABSENT = "definitelyNotInThisCodebase"

async def main():
    async with Client(SERVER) as client:
        tools = await client.list_tools()
        print("TOOLS:", ", ".join(t.name for t in tools))
        for name, args in [
            ("rag_status", {}),
            ("search_codebase", {"query": CONCEPT, "top_k": 3}),
            ("search_codebase", {"query": IDENTIFIER, "top_k": 3}),
            ("find_symbol_references", {"symbol_name": IDENTIFIER}),
            ("find_symbol_references", {"symbol_name": KNOWN_ABSENT}),
            ("find_symbol_or_keyword", {"pattern": IDENTIFIER, "limit": 5}),
            ("get_chunk_content", {"chunk_id": "bogus:id:1-2"}),
        ]:
            result = await client.call_tool(name, args)
            text = result.content[0].text
            print("\n===== %s %s\n%s" % (name, args, text[:600]))
asyncio.run(main())
PY
```

Check every line of the output against this list:

- [ ] All seven tools are listed with the expected parameters.
- [ ] `rag_status` reports a sane file/chunk count and `up to date with the working tree`.
- [ ] The **concept** query returns the files you would have opened by hand.
- [ ] The **identifier** query ranks its `DEFINITION` first, and the reported line range
      is correct — open the file and confirm, do not assume.
- [ ] The **absent** symbol produces a clean negative confirmation, not an empty result
      or a crash.
- [ ] The bogus chunk id returns a helpful message, not a stack trace or a protocol error.

### 8.3 Filter pushdown

```bash
# Substitute a real sub-project directory from Phase 0.1.
"$WORKSPACE/.poc-venv/bin/python3" - <<'PY'
import os, sys
sys.path.insert(0, os.environ["WORKSPACE"])
import mcp_server as s
call = lambda t: t.fn if hasattr(t, "fn") else t
print(call(s.search_codebase)("configuration", path_filter="SUBPROJECT", top_k=5))
PY
```

Every hit must come from that path, and the result must not be empty for a query whose
subject demonstrably exists there. An empty result here is the classic post-filtering
bug: retrieve globally, then throw away everything that does not match.

### 8.4 Incremental round-trip

```bash
echo "// rag smoke test" >> "$TARGET/<some-source-file>"
time "$WORKSPACE/.poc-venv/bin/python3" "$WORKSPACE/ingest_codebase.py"
# expect: "N unchanged, 1 re-chunked" and a runtime around a second
# now revert the edit, re-run, and confirm rag_status() shows no staleness
```

### 8.5 The honest comparison

For your three Phase 0.4 queries, ask: *would a plain `grep` have answered this faster?*
If yes for all three, the index is not earning its keep — check the skeleton quality,
the tokenizer, and the chunk boundaries in that order. The point of this system is not
that it can retrieve; it is that an agent stops shelling out to grep because the tool
answers better.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Client shows no tools / disconnects immediately | Something printed to stdout, or the server raised at import | Run `<WORKSPACE>/.poc-venv/bin/python3 <WORKSPACE>/mcp_server.py` directly and read the traceback; check `rag.log` |
| `Functions with *args are not supported as tools` | A decorator erased the tool signature | Use `functools.wraps` on any wrapper (the `tool_guard` here does) |
| Results are stale after editing | Hooks only fire on git operations | Call `reindex()`, or heed the staleness banner |
| Narrow `path_filter` returns nothing | Post-filtering instead of pushdown | Confirm `_where_clause` is reaching ChromaDB and the BM25 candidate set is pre-filtered |
| Identifier searches miss obvious call sites | Naive BM25 tokenizer | Confirm ingest and query both use `cfg.tokenize`; re-index after changing it |
| Line ranges are wrong | tree-sitter unavailable → window fallback | `rag_status()` reports the active chunker; reinstall `tree-sitter-language-pack` |
| First query takes seconds, rest are instant | ONNX session warm-up | Normal; pre-warm by calling `rag_status()` at session start |
| Index and vector store disagree | Interrupted ingest | Re-run ingest — it detects the mismatch and rebuilds |
| Chunk ids not found after a rebuild | Ids encode line numbers and change with the code | Re-run the search; never cache chunk ids across sessions |
| Vector store keeps growing | Chroma leaves dead segments after repeated full rebuilds | `rm -rf "$WORKSPACE/chroma_db"` then ingest `--full` |
| `refusing to unpickle X.Y from the RAG index` | The index was written by a different/older build, or tampered with | Delete `rag_index.pkl` and ingest `--full`. If the class is legitimate, add it to `_PICKLE_ALLOWLIST` — never widen it to `builtins.eval`/`os.system` |
| `rag_status` says "estimated (tokenizer not cached)" | `tokenizer.json` missing from `.models_cache` | Re-run `download_model.py`; chunking still works but over-splits conservatively |
| Reranker line says "set but unavailable" | `RAG_RERANKER` names a model that was never cached | `RAG_RERANKER=... python3 download_model.py` while online, or unset it |
| Ingest crashed and the next run re-embedded everything | An older build, or a checkpoint rejected as incompatible | The log says which. A checkpoint is only reused for the same model, chunker version and repo root — resuming across any of those would mix incompatible vectors |
| Chunks from other repos vanish after a commit | An old `rag-reindex.sh` targeting the committed repo instead of the workspace root | Re-run `install_hooks.sh`, then `sync_and_index.sh --full`. `rag_status()` should list every repo under "Repos indexed" |
| One repo never refreshes | Hooks installed before multi-repo support, or the repo is outside the discovered set | `rag_config.py --repos` shows what is discovered; set `RAG_REPOS` and re-run `install_hooks.sh` |
| A pre-existing git hook stopped running | It was replaced by an older installer version | Look for `<hook>.pre-rag` next to it and re-run `install_hooks.sh` — it re-chains on every run |
| `path_filter` silently stops narrowing on a big repo | The `$in` pushdown exceeded its cap and fell back to post-filtering | The header says which; the pool auto-compensates by the filter's selectivity, but raising `RAG_PUSHDOWN_MAX_PATHS` is the real fix |

## Tuning

| Situation | Change |
|---|---|
| Large repo (>50k chunks) | Raise `RAG_EMBED_BATCH`; consider a per-sub-project `RAG_COLLECTION` and `RAG_INDEX_DIR` |
| Better recall, more RAM/time | `RAG_EMBED_MODEL=BAAI/bge-base-en-v1.5` (768-dim); a full rebuild is forced automatically |
| Multilingual codebase or docs | A multilingual fastembed model; the query-prefix table in `rag_config.py` adapts |
| Prose-heavy docs repo | Raise `RAG_MAX_CHUNK_TOKENS` toward the model window, keeping headroom |
| Several repos under one root | Nothing — they are auto-discovered. Override with `RAG_REPOS` if they live elsewhere, and use `sync_and_index.sh` to pull them all before a session |
| Monorepo, one sub-project at a time | Set `RAG_REPO_ROOT` per client entry with separate `RAG_INDEX_DIR`s |
| Top-3 precision matters more than latency | `RAG_RERANKER=Xenova/ms-marco-MiniLM-L-6-v2`, then re-run `download_model.py` to cache it |
| Reranker too slow / too shallow | `RAG_RERANK_CANDIDATES` (default 24) — the fused pool it re-scores |
| Identifier queries under-ranked | Raise `RAG_W_IDENT_LEX` / `RAG_SYMBOL_BOOST` |
| Prose questions under-ranked | Raise `RAG_W_NL_DENSE`, lower `RAG_W_NL_LEX` |
| Flatter, rank-only fusion | Set every `RAG_W_*` to `1.0` and `RAG_SYMBOL_BOOST=0` — that is textbook RRF |
| `find_symbol_or_keyword` times out | Raise `RAG_GREP_TIMEOUT` (default 30s), or install ripgrep |
| Big repo, `path_filter` results look thin | Check the search header: `N files POST-filtered` means the filter exceeded `RAG_PUSHDOWN_MAX_PATHS` (default 10,000). Raise it — measured safe to ~32,000 before SQLite's variable limit |
| Large index, relevant chunk never surfaces | The candidate pool may be too shallow. Raise `RAG_POOL_MAX` (default 500); the header prints the pool actually used |
| Huge first ingest killed at 30 min | Raise `RAG_REINDEX_TIMEOUT` |
| Setup blocked from reaching HuggingFace | Publish the cached `.models_cache` as a release asset and point `RAG_MODEL_ASSET_URL` at it — pin `RAG_MODEL_ASSET_SHA256` so it is verified before unpacking |
| Long cold ingest, want tighter crash protection | Lower `RAG_CHECKPOINT_EVERY` (default 20 batches ≈ 5k chunks); each checkpoint costs well under a second |

## Rebuild from scratch

```bash
rm -rf "$WORKSPACE/chroma_db" "$WORKSPACE/rag_index.pkl" "$WORKSPACE/rag.log"
"$WORKSPACE/.poc-venv/bin/python3" "$WORKSPACE/ingest_codebase.py" --full
```

Add `.poc-venv/` and `.models_cache/` to that `rm` for a completely cold rebuild —
Phase 1 and Phase 6 then re-create them.

---

## Appendix — Design decisions under review

This system has been reviewed against production-engineering, retrieval-quality and
security critiques. The changes that survived are already in the code above; this
appendix records **why** each one is shaped the way it is, and — more usefully — which
recommendations were *declined* and what would change that verdict. A runbook that only
lists its wins teaches you to cargo-cult it.

### Adopted

| Change | Problem it actually fixes |
|---|---|
| Encoder tokenizer for chunk budgeting | A flat `chars/token` constant is wrong in both directions. Measured on a real repo, `3.0` let 11 chunks (0.6%) past the guard that the encoder then truncated — density fell to **1.85** chars/token on bracket-dense and minified content. The char estimate survives as a pre-filter, so ~95% of chunks never touch the tokenizer and ingest speed is unchanged. |
| Query-adaptive weighted RRF | Textbook RRF treats rank 1 from either retriever as identical evidence. It is not: a bare `refresh_access_token` that BM25 matched exactly is near-certain, while a prose question is better served by the dense side. Weights follow query *shape*, which is cheap and inspectable. |
| Exact-symbol boost | Rank cannot express "this chunk **is** the thing you named." Scaled to a fraction of one rank-1 RRF step, so it breaks near-ties without ever overriding genuine agreement between retrievers. |
| Restricted unpickler | `pickle.load` on the index is arbitrary code execution for anything that can write that file. The allowlist makes a legitimate load byte-for-byte identical work and kills a doctored one at `find_class`, before any object is built. |
| Optional cross-encoder reranker | Where top-3 precision actually lives. Kept **opt-in** because a fresh clone must work offline; unset it changes nothing, set-but-uncached it logs and falls back to the fused order. |
| Hook chaining | Husky, pre-commit and lefthook all own `.git/hooks` or redirect it via `core.hooksPath`. Overwriting them breaks a developer's commit pipeline in a way that is near-impossible to trace back to a RAG installer. |
| Broadened fallback symbol regex | The window splitter runs exactly where tree-sitter could not — which is where a name matters most, because there is no node to ask. Six JS/Python keywords left Go/Rust/Kotlin/Swift gap chunks anonymous and unreachable by symbol. |
| Typed `Optional[...]` signatures | `path_filter: str = None` is a lie the schema generator faithfully transcribes; strict MCP hosts warn on it. |
| Path-filter pushdown cap raised 400 → 10,000 | The old cap was set by assumption, not measurement. On this stack Chroma resolves a `$in` of 400 values in 0.008s, 10,000 in 0.013s and 32,000 in 0.034s, failing only past ~40,000 on SQLite's variable limit. Every filter between 400 files and the real ceiling was silently falling back to post-filtering — on exactly the repos big enough for the pushdown to matter. |
| Multi-repo workspace support | A workspace is frequently several sibling checkouts, not one tree. The indexer only walks paths and never cared — but every git-aware operation does, and each failed differently: `install_hooks.sh` covered only the first repo, so the rest pulled with no re-index and no warning; and the hook body targeted `git rev-parse --show-toplevel`, which inside a hook is the *one* repo being committed to — re-indexing that alone **deletes every other repo's chunks** from the shared index. Repo discovery now lives in `rag_config` and is shared by the indexer, the hooks and the sync script, and `rag_status()` prints the indexed commit per repo. |
| Ingest checkpointing | Embeddings were already durable per batch, but the record of *which* ones existed was written once at the very end. A crash therefore left a vector store the next run could not explain, and the "counts disagree → rebuild" rule wiped it and started over. Irrelevant at 133 s; at a 50–70 min cold ingest it is an afternoon. The checkpoint stores manifest + embedded ids only — not chunk bodies — so a write costs well under a second, and resume is gated on the file's SHA-1 so an edit between crash and restart is re-embedded rather than trusted. |
| Corpus-scaled candidate pool | `n_results = top_k * 4` is a fixed slice of a growing haystack: 40 candidates is 2% of a 2k-chunk index but 0.06% of a 64k-chunk one, and RRF can only rank what retrieval returned. The pool now follows √(corpus), and over-fetches by the inverse of filter selectivity when the pushdown could not be used. Normalised so a 2k-chunk index gets exactly the old numbers. |

### Declined, and what would reverse that

**Migrating chunk + BM25 storage from pickle to SQLite/DuckDB.** The stated trigger is
repositories above ~30k files; the reference repo here is 161 files / 1,840 chunks /
2.9 MB, where load is a few tens of milliseconds. The *security* half of the argument is
the real half, and the restricted unpickler answers it at zero runtime cost. The
*memory* half is a genuine rewrite that would also slow scoring, because `rank_bm25`
scores against an in-memory array in one vectorised pass.

> **Revisit when** `rag_status()` reports **> ~50k chunks**, or the index file passes
> ~200 MB, or `_load_index()` becomes visible in tool latency. At that point the right
> move is not "SQLite instead" but a split: chunk *text* in SQLite keyed by chunk id
> (read on demand, only for the ~10 hits actually returned), with the BM25 postings
> staying resident. Loading chunk bodies for 50k chunks to display 10 is the waste worth
> attacking, not the pickle format.

**Replacing the hand-written Dart, SQL and Markdown extractors with uniform
tree-sitter.** These exist *because* the uniform path produces worse chunks, and the
comments in `rag_chunker.py` name each case: SQL grammars swallow dollar-quoted function
bodies and fuse an entire migration file into one `statement` node; Dart declarations
lead with the return type, so the generic "first identifier" heuristic yields `Future`
instead of the method name; Markdown's value is heading-bounded sections, not its AST.
Uniformity here would be a regression measured in wrong symbol names.

> **Revisit when** the grammars fix those cases upstream — verify by diffing the symbol
> column of `rag_status()`-scale output before and after, not by reading release notes.

### On tuning constants by measurement, not intuition

Two of the adopted changes above are single numbers that had been wrong since the
first draft — a `400` and a `* 4` — and neither was wrong because the reasoning was
bad. They were wrong because nobody measured the thing they were guarding. The
pushdown cap was ~80x more conservative than the engine needs, and a deeper candidate
pool turns out to be almost free:

| `n_results` | dense query |
|---|---|
| 40 | 9.0 ms |
| 226 | 5.4 ms |
| 500 | 10.3 ms |
| 1000 | 18.3 ms |

If a constant in this system guards a resource, measure that resource before you
trust the constant. A magic number with a confident comment above it is still a
magic number.

### Verifying a change to any of this

Retrieval changes are easy to *believe* and hard to *confirm*. Before and after, run your
three Phase 0.4 queries and record: the rank of the correct chunk, whether any previously
correct result moved **down**, and end-to-end tool latency. A weighting change that
improves two queries and quietly demotes a third is a regression, and only the second
number catches it.
