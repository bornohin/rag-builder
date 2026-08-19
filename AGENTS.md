# Agent instructions

What an AI assistant needs to know to actually use this retrieval server: the seven
tools, when to reach for each, and how index freshness works. Nothing below is
specific to any project or any assistant.

**This has to reach your assistant's ROOT instruction file.** Assistants load the
project-root file at session start and a subdirectory file only when they touch that
subtree — so this file sitting here, unreferenced, is read exactly when it is not
needed and stays silent when it is. Two ways to wire it up:

**Import it** (preferred — one copy, so it cannot drift). In your project-root
`CLAUDE.md` / `GEMINI.md` / `AGENTS.md`:

```markdown
@rag-workspace/AGENTS.md
```

**Or paste it** if your assistant has no import syntax. Copy everything below the
rule into that root file, and re-copy it after a `git pull` that changes this file.
Two copies drift, and the stale one is the one that gets followed.

---

## Local Codebase RAG (MCP server `codebase-rag`)

100% local and offline: tree-sitter chunking → ONNX embeddings → ChromaDB (dense)
+ BM25 (lexical) → query-adaptive weighted Reciprocal Rank Fusion. Every hit
reports an **exact `file:start_line-end_line`** you can act on directly. Plain
stdio MCP with no vendor-specific configuration, so the same setup serves any
assistant.

Fusion weights follow the shape of the query (a bare identifier trusts BM25; a
prose question trusts the dense side) and a chunk whose own symbol is a query term
earns a bounded boost. An optional local cross-encoder reranker is available but
**off by default** — see `rag-workspace/README.md` for the `RAG_*` knobs.

### Tools

1. **`search_codebase(query, category="source_code", path_filter=None, top_k=10, return_skeletons=True)`**
   - Start here for any "where / how is X implemented?" question. Natural language
     and bare identifiers both work — BM25 is tokenized on snake_case and camelCase.
   - `category`: `'source_code'` (default), `'documentation'`, `'config'`, `'all'`.
   - `path_filter`: path substring. Pushed down into both retrievers, so narrowing
     never starves the result set.
   - `return_skeletons=True` returns the signature plus the first meaningful body
     lines; pass `False`, or call `get_chunk_content(chunk_id)`, for the full body.

2. **`find_symbol_references(symbol_name, path_filter=None, category="source_code", limit=10)`**
   - Definitions and call sites for a function, class, table, column, or RPC name.
     Word-boundary matched; results labelled `DEFINITION` / `call site` / `mention`
     and ranked in that order. A "zero references" answer is confirmed against a
     live grep of the working tree, so a stale index cannot produce a false negative.

3. **`find_symbol_or_keyword(pattern, path_filter=None, file_pattern="*", limit=40, offset=0)`**
   - Literal/regex search of the working tree — always current, never stale.
     Paginated: when output is truncated, call again with the `offset` in the footer.

4. **`get_file_context(targets=[{"path": ..., "start_line": N, "end_line": M}])`**
   - Batch reader. Pull every range you need across every file in **one** call.

5. **`get_chunk_content(chunk_id)`** — full verbatim source behind a search hit.

6. **`rag_status()`** — chunk/file counts, model, build time, per-repo indexed
   commits, fusion constants, candidate-pool depth, and which files are stale.

7. **`reindex(full=False)`** — refresh after edits. Incremental (~1s); `full=True`
   only after changing the embedding model or chunker.

`search_codebase` prints the candidate pool it used and whether a `path_filter` was
pushed into the vector store or post-filtered. `POST-filtered` in that header means
the filter exceeded `RAG_PUSHDOWN_MAX_PATHS` and recall is being propped up by
over-fetching.

### Workflow directives

- **Search before shell.** Use `search_codebase` / `find_symbol_references` /
  `find_symbol_or_keyword` instead of `grep`, `find`, or sequential file reads.
- **Batch your reads.** One `get_file_context` call with several targets beats
  several single-file reads.
- **Trust the line numbers.** Chunk boundaries are tree-sitter nodes, so
  `start_line-end_line` is the symbol itself, not a sliding window.
- **Heed the staleness banner.** If a result is tagged `[index staleness]`, call
  `reindex()` before drawing conclusions. Git hooks re-index automatically after
  pull/commit/checkout/rebase, but not after uncommitted edits.
- **Keep answers focused.** Lead with file paths, line numbers, and the finding.

### Freshness

- **Session start / manual**: `rag-workspace/sync_and_index.sh` pulls **every** repo
  in the workspace (skipping any with uncommitted work, a detached HEAD or no
  upstream), then refreshes the index once. This is the command to run before
  starting work; a `SessionStart` hook can run the same thing.
- **Git operations**: `post-merge` / `post-commit` / `post-checkout` / `post-rewrite`
  hooks re-index incrementally, in every repo `install_hooks.sh` found.
- **Uncommitted edits**: not covered by either — that is what the `[index staleness]`
  banner and `reindex()` are for.

### Maintenance

```bash
rag-workspace/sync_and_index.sh                                     # pull + incremental index
rag-workspace/.poc-venv/bin/python3 rag-workspace/ingest_codebase.py         # incremental only
rag-workspace/.poc-venv/bin/python3 rag-workspace/ingest_codebase.py --full  # rebuild
rag-workspace/install_hooks.sh                                      # (re)install git hooks
```

Set `RAG_SKIP_PULL=1` to refresh the index without touching git.

---

## Standards worth keeping

1. **LLM-agnostic tools.** Skills, scripts and MCP configuration stay portable across
   assistants — stdio transport, env-var configuration, no vendor-specific fields.
2. **Validation.** Test and verify changes empirically before calling a task done.
3. **Offline by default.** Model weights and indexing stay local and in user space.
4. **Measure the constants.** Batch sizes, result limits and safety caps guard real
   resources — profile the resource before trusting the number. A confident comment
   above a guessed value is still a guess.
