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
import pickle
import re
import subprocess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _detect_repo_root() -> str:
    """RAG_REPO_ROOT env > enclosing git worktree > parent of rag-workspace."""
    env = os.environ.get("RAG_REPO_ROOT")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    try:
        out = subprocess.check_output(
            ["git", "-C", BASE_DIR, "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        root = out.decode().strip()
        if root and os.path.isdir(root):
            return root
    except Exception:
        pass
    return os.path.dirname(BASE_DIR)


REPO_ROOT = _detect_repo_root()
INDEX_DIR = os.path.abspath(os.environ.get("RAG_INDEX_DIR", BASE_DIR))
CACHE_DIR = os.path.join(INDEX_DIR, ".models_cache")
CHROMA_DIR = os.path.join(INDEX_DIR, "chroma_db")
INDEX_PATH = os.path.join(INDEX_DIR, "rag_index.pkl")      # chunks + bm25 + manifest
LEGACY_BM25_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")
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


# Hard ceiling of the encoder. Chunks longer than this are silently truncated
# by the tokenizer, so the chunker splits before reaching it.
MODEL_MAX_TOKENS = int(os.environ.get("RAG_MODEL_MAX_TOKENS", "512"))
MAX_CHUNK_TOKENS = int(os.environ.get("RAG_MAX_CHUNK_TOKENS", "440"))
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH", "256"))

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


def load_index_file(path: str = None) -> dict:
    """Load the index through the restricted unpickler. Never raises."""
    path = path or INDEX_PATH
    if not os.path.exists(path):
        return dict(EMPTY_INDEX)
    with open(path, "rb") as fh:
        data = RestrictedUnpickler(fh).load()
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
