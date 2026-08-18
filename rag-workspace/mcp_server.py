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
