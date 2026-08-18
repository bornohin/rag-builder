# rag-builder

Build a **100% local, offline, LLM-agnostic hybrid codebase RAG server** over any
repository, served to any AI assistant over stdio MCP.

Tree-sitter structure-aware chunking → ONNX embeddings on the CPU (ChromaDB) +
code-aware BM25 → query-adaptive weighted Reciprocal Rank Fusion. Every hit reports an
**exact `file:start_line-end_line`**, so an agent can act on the answer instead of
going back to `grep`.

No sudo, no daemon, no vendor lock-in, and no network at query time.

## What's here

| Path | What it is |
|---|---|
| **`create_rag.md`** | The runbook. A complete, self-contained build guide — survey the target repo, write each file, wire up the client, verify. Hand it to an agent or follow it yourself. |
| `rag-workspace/` | The reference implementation the runbook produces. Copy it into the repo you want to index, or build it from the runbook. |
| `skills/rag/SKILL.md` | An interactive, one-step-at-a-time setup skill for agents that support skills. |
| [`docs/architecture.html`](docs/architecture.html) | A ten-slide visual explainer of how the whole thing works — the pipeline, chunking, hybrid search, rank fusion, and where it breaks at scale. Open it in a browser. |

## Quickstart

Copy `rag-workspace/` into the root of the repository you want to index, then:

```bash
python3 -m venv rag-workspace/.poc-venv
rag-workspace/.poc-venv/bin/pip install -r rag-workspace/requirements.txt
rag-workspace/.poc-venv/bin/python3 rag-workspace/download_model.py
rag-workspace/.poc-venv/bin/python3 rag-workspace/ingest_codebase.py --full
rag-workspace/install_hooks.sh          # optional: keep the index fresh on git ops
```

Register it with any MCP-capable client as a stdio server:

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

Full tool reference, every `RAG_*` environment variable, design notes and
troubleshooting live in [`rag-workspace/README.md`](rag-workspace/README.md).

## A word on the generated index

`rag_index.pkl` and `chroma_db/` store **verbatim source text** of whatever you
indexed. They are gitignored here and should stay that way — committing them
publishes the code you indexed. They rebuild from scratch in one command.

## Requirements

Python 3.9+, `git` (optional), ~500 MB of disk for the virtualenv and model weights,
and one network fetch during setup to install packages and cache the model. After
setup, nothing phones home.

## License

MIT — see [LICENSE](LICENSE).
