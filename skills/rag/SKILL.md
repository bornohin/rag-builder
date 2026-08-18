---
name: rag
description: Interactive setup guide for deploying a 100% local, LLM-agnostic hybrid codebase RAG system using FastMCP, ChromaDB, FastEmbed (ONNX), tree-sitter and BM25. Use when building, configuring, running, or troubleshooting a local codebase RAG indexing and search pipeline.
---

# Local Hybrid Codebase RAG — Setup Guide

You are an interactive setup assistant deploying a 100% local, offline, policy-compliant
hybrid codebase RAG system: **tree-sitter** chunking, **FastEmbed** (ONNX, CPU) dense
embeddings, **ChromaDB** vector store, **BM25** lexical index, **Reciprocal Rank Fusion**,
served over **FastMCP** stdio so any MCP-capable assistant can use it.

Follow these strict rules:
1. Guide the user ONE step at a time; wait for confirmation before the next step.
2. Prompt for paths before generating or running anything.
3. Keep execution 100% offline, in user-space, no sudo, no network calls at query time.
4. Never emit assistant-specific configuration — stdio transport and environment
   variables only, so the result works with Claude Code, Gemini/Antigravity, Cursor,
   Windsurf, Continue, Zed, or a custom client.

A complete reference implementation of every file below already exists in this
repository at `rag-workspace/` — read it before writing anything new.

---

### Step 1: Workspace & Target Repo
Ask:
1. "Where should the RAG workspace live? (Default: `<repo>/rag-workspace`)"
2. "Which repository should be indexed? (Default: the enclosing git worktree)"

---

### Step 2: Environment & Dependencies
Create `.poc-venv` and install from `requirements.txt`:
`fastmcp`, `chromadb`, `fastembed`, `rank-bm25`, `tree-sitter`, `tree-sitter-language-pack`.
Confirm the install succeeded before continuing.

---

### Step 3: Cache the Embedding Model
Generate `download_model.py`, which reads its model name from `rag_config.py`
(default `BAAI/bge-small-en-v1.5`, 384-dim, 512-token window, Apache/MIT-licensed
ONNX weights) and caches it into `./.models_cache`. Confirm the download completes —
after this, nothing touches the network.

---

### Step 4: Generate the Pipeline
Write four modules, in this order:

1. **`rag_config.py`** — the single source of truth. Paths, model name, exclusion
   rules, chunk-token budget, and the shared tokenizer. Every value overridable by
   environment variable (`RAG_REPO_ROOT`, `RAG_INDEX_DIR`, `RAG_EMBED_MODEL`,
   `RAG_MAX_CHUNK_TOKENS`, `RAG_EMBED_BATCH`, `RAG_COLLECTION`). Duplicating any of
   these into another file is how a RAG pipeline silently drifts out of sync.

2. **`rag_chunker.py`** — structure-aware chunking. Non-negotiables:
   - Chunk on **tree-sitter nodes**, not sliding windows, so `start_line`/`end_line`
     describe the symbol itself. Degrade to a heading/window splitter if tree-sitter
     is unavailable, never fail.
   - **Respect the encoder's token window.** Split oversized symbols into labelled
     `part i/n` pieces; anything longer is silently truncated at embed time.
   - Decompose a too-large symbol into its *structural children* before falling back
     to a line split, so a 1200-line component yields its real inner functions.
   - **Gap-fill**: window-index any line the parser did not claim, so recall is never
     worse than a naive index.
   - Skeletons must show the **signature plus the first meaningful body lines** — not
     the first N raw lines, which for a window is usually a closing brace.

3. **`ingest_codebase.py`** — incremental indexer. Maintain a SHA-1 manifest per file;
   re-chunk and re-embed only what changed; delete vectors for removed files; stream
   embeddings in batches rather than materializing them all; write the index
   atomically (`tmp` + `os.replace`). Force a full rebuild when the embedding model or
   the chunker version changes, or when the vector store and index disagree.

4. **`mcp_server.py`** — the FastMCP stdio server. Non-negotiables:
   - Apply the model's **query instruction prefix** to queries only (bge/e5/arctic are
     asymmetric); passages stay bare.
   - Use a **code-aware BM25 tokenizer** shared with ingestion — identifiers indexed
     whole *and* split on snake_case/camelCase, otherwise
     `supabase.rpc('delete_ephemeral_message')` never matches that identifier.
   - **Push filters down** into ChromaDB (`$and` / `$in`) and BM25 rather than dropping
     hits from the result list, so a narrow `path_filter` cannot starve a query.
   - **Hot-reload** the index when its mtime changes; no restart after a re-index.
   - **Report staleness** on every result, and confirm every negative answer against a
     live grep of the working tree.
   - Log to a file — **stdout belongs to the MCP transport**.
   - Wrap each tool so an exception returns a readable message instead of killing the
     session.

Expose exactly these tools: `search_codebase`, `find_symbol_references`,
`find_symbol_or_keyword` (paginated), `get_chunk_content`, `get_file_context`,
`rag_status`, `reindex`.

Then run the initial index and report the chunk count, per-language breakdown, and
build time.

---

### Step 5: Wire the Client & Git Hooks
1. Register the server with any MCP host as a stdio server:
   ```json
   {"mcpServers": {"codebase-rag": {"type": "stdio",
     "command": "<workspace>/.poc-venv/bin/python3",
     "args": ["<workspace>/mcp_server.py"], "env": {}}}}
   ```
2. Run `install_hooks.sh` to install `post-merge`, `post-commit`, `post-checkout` and
   `post-rewrite` hooks that re-index incrementally (typically under two seconds).
3. Add the index artifacts (`chroma_db/`, `rag_index.pkl`, `.models_cache/`,
   `.poc-venv/`, `rag.log`) to `.gitignore`.
4. Verify end to end: call `rag_status()`, then a `search_codebase` query, and confirm
   the returned line ranges match the real source.

---

### Step 6: Validate Retrieval Quality
Before declaring success, check all four — these are the failure modes that make a RAG
server look functional while being useless:
1. Search a known identifier; the **definition** must rank first with correct lines.
2. Search with a narrow `path_filter`; results must come from that path, not be empty.
3. Confirm no chunk exceeds the encoder's token window.
4. Edit a file, re-index, and confirm the change is reflected and the staleness banner
   clears.
