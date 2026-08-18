# Local Hybrid Codebase RAG

A 100% local, offline, LLM-agnostic code retrieval server. It indexes this
repository with tree-sitter, embeds chunks with an ONNX model that runs on the
CPU, and serves hybrid (dense + BM25) search over stdio MCP — so any
MCP-capable assistant (Claude Code, Gemini/Antigravity, Cursor, Windsurf,
Continue, Zed, a custom client) gets the same tools with no vendor-specific
configuration.

## Layout

| File | Role |
|---|---|
| `rag_config.py` | Single source of truth: paths, model, exclusions, tokenizer. Everything overridable by env var. |
| `rag_chunker.py` | Tree-sitter structure-aware chunking with exact line ranges, token-budget splitting, gap filling. |
| `ingest_codebase.py` | Incremental indexer (SHA-1 manifest) → ChromaDB + BM25 → `rag_index.pkl`. |
| `mcp_server.py` | FastMCP stdio server exposing the 7 tools. |
| `download_model.py` | One-time model cache warm-up (embedder, tokenizer, and the optional reranker). |
| `install_hooks.sh` | Installs git hooks that re-index after pull/commit/checkout/rebase, chaining any hooks already present. |

## Setup

```bash
python3 -m venv rag-workspace/.poc-venv
rag-workspace/.poc-venv/bin/pip install -r rag-workspace/requirements.txt
rag-workspace/.poc-venv/bin/python3 rag-workspace/download_model.py
rag-workspace/.poc-venv/bin/python3 rag-workspace/ingest_codebase.py --full
rag-workspace/install_hooks.sh
```

Register with any MCP client as a stdio server:

```json
{
  "mcpServers": {
    "codebase-rag": {
      "type": "stdio",
      "command": "<repo>/rag-workspace/.poc-venv/bin/python3",
      "args": ["<repo>/rag-workspace/mcp_server.py"],
      "env": {}
    }
  }
}
```

## Tools

| Tool | Use it for |
|---|---|
| `search_codebase(query, category, path_filter, top_k, return_skeletons)` | Any "where/how is X implemented" question. Hybrid dense + BM25, RRF-fused. Returns exact `file:start-end`. |
| `find_symbol_references(symbol_name, path_filter, category, limit)` | Definitions and call sites of an identifier. Word-boundary matched, definitions ranked first, negatives verified against the live tree. |
| `find_symbol_or_keyword(pattern, path_filter, file_pattern, limit, offset)` | Literal/regex grep of the working tree. Always current, paginated. |
| `get_chunk_content(chunk_id)` | Full body behind a search hit. |
| `get_file_context(targets)` | Exact line ranges from many files in one call. |
| `rag_status()` | Chunk counts, model, build time, and which files are stale. |
| `reindex(full=False)` | Refresh after edits — incremental, usually under two seconds. |

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `RAG_REPO_ROOT` | enclosing git worktree | Index a different repository. |
| `RAG_INDEX_DIR` | `rag-workspace/` | Where the index artifacts live. |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed-supported model (query prefix adapts automatically). |
| `RAG_MAX_CHUNK_TOKENS` | `440` | Chunk budget; keep below the model's window. |
| `RAG_EMBED_BATCH` | `256` | Embedding batch size. |
| `RAG_COLLECTION` | `codebase` | ChromaDB collection name. |
| `RAG_RERANKER` | *(unset)* | Cross-encoder for top-3 precision, e.g. `Xenova/ms-marco-MiniLM-L-6-v2`. Off by default so a fresh clone stays offline; cache it with `download_model.py` before enabling. |
| `RAG_RERANK_CANDIDATES` | `24` | How many fused candidates the reranker re-scores. |
| `RAG_RRF_K` | `60` | RRF damping constant. |
| `RAG_SYMBOL_BOOST` | `0.6` | Bonus for a chunk whose own symbol is a query term, as a fraction of one rank-1 RRF step. `0` disables it. |
| `RAG_W_IDENT_DENSE` / `RAG_W_IDENT_LEX` | `0.7` / `1.4` | Fusion weights for identifier-shaped queries. |
| `RAG_W_NL_DENSE` / `RAG_W_NL_LEX` | `1.3` / `0.8` | Fusion weights for natural-language queries. |
| `RAG_W_MIX_DENSE` / `RAG_W_MIX_LEX` | `1.0` / `1.0` | Fusion weights for everything else. |
| `RAG_GREP_TIMEOUT` | `30` | Seconds before the live grep gives up. |
| `RAG_REINDEX_TIMEOUT` | `1800` | Seconds before `reindex()` gives up. |

Setting every `RAG_W_*` to `1.0` and `RAG_SYMBOL_BOOST=0` reduces fusion to
textbook RRF — useful as a baseline when measuring a retrieval change.

## Design notes

- **Exact line ranges.** Chunks are tree-sitter nodes, not sliding windows, so
  `start_line`/`end_line` point at the symbol itself.
- **Nothing is silently truncated.** Chunk sizes are measured with the encoder's
  *own* tokenizer, not a chars-per-token constant — real code ranges from ~1.8
  (minified, bracket-dense) to ~4.0 (prose) chars/token, and a single constant
  cannot serve both. Oversized symbols become labelled `part i/n` pieces, and
  every piece is verified against the tokenizer before it is emitted.
- **Code-aware BM25.** Identifiers are indexed whole *and* split on snake_case
  and camelCase, so `refresh_access_token` matches
  `client.rpc('refresh_access_token', …)`.
- **Asymmetric query embedding.** The bge/e5/arctic instruction prefix is
  applied to queries only, as those models were trained.
- **Filters are pushed down** into ChromaDB and BM25 rather than applied to the
  result list, so a narrow `path_filter` can never starve a query.
- **Fusion knows what kind of question it was asked.** RRF alone treats rank 1
  from either retriever as identical evidence. A bare identifier that BM25
  matched exactly is stronger evidence than a middling vector neighbour, and a
  prose question is the reverse — so the weights follow the query's shape, and
  a chunk whose own symbol is a query term earns a bounded bonus on top.
- **The reranker is optional and silent when absent.** Unset, nothing changes.
  Set but uncached, searches still answer and the reason is in `rag.log`.
- **The index is loaded through a restricted unpickler.** Only the handful of
  classes a real index contains will resolve; anything else raises before a
  single object is constructed, so a tampered index cannot run code in the
  server. `reindex(full=True)` is the recovery path.
- **Git hooks chain, they do not clobber.** `core.hooksPath` is honoured (that
  is how husky redirects hooks), any pre-existing hook is preserved as
  `<hook>.pre-rag` and still runs first, and indexing can never fail the git
  operation the user actually asked for.
- **Staleness is surfaced, never hidden.** Every result carries a warning when
  indexed files have changed on disk, and negative answers are confirmed
  against a live grep.
