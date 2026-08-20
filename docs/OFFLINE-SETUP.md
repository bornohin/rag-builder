# Setting up where HuggingFace is blocked

Written for a corporate laptop that can reach GitHub and PyPI but not
HuggingFace. Nothing here is a workaround: the model is published as a release
asset on this repository, and `download_model.py` prefers it by default. There
is no configuration to set.

The only external dependency left is PyPI, for `pip install`.

---

## The model asset

| | |
|---|---|
| **URL** | `https://github.com/bornohin/rag-builder/releases/download/models-v1/bge-small-en-v1.5-onnx-q.tar.gz` |
| **SHA-256** | `3830f02a8ccbfb88dfe2cb460e2e5c6e2e1ac9e106f60c836e7ecf7b4cb42187` |
| Size | 58 MB compressed, 64 MB unpacked |
| Auth | none — public, unauthenticated |
| Model | `BAAI/bge-small-en-v1.5`, Qdrant's quantized ONNX build, revision `52398278842ec682c6f32300af41344b1c0b0bb2` |

**Allowlist note.** That URL 302-redirects to `https://release-assets.githubusercontent.com`.
Some proxies permit `github.com` but not the object CDN, which shows up as a
redirect that dies rather than a clean refusal. If `github.com` is allowed and the
download still fails, that host is the thing to get allowlisted — it is the same
shape as networks that permit `pypi.org` but block `files.pythonhosted.org`.

---

## Instructions for an AI agent

Paste this verbatim.

```text
Set up a local code-search MCP server for this project. It runs entirely on this
machine — no HuggingFace, no external model API. Our corporate network blocks
HuggingFace; the model is served from our own GitHub release instead, and the
setup script already knows that. Do NOT work around HuggingFace being blocked —
it is not needed.

From the root of the project you want to make searchable:

  # 1. The repo IS the workspace — clone it in under this exact name.
  git clone https://github.com/bornohin/rag-builder.git rag-workspace

  # 2. Create the venv with an EXPLICIT interpreter. Python 3.10+ is required.
  #    macOS ships /usr/bin/python3 as 3.9, and pip's failure ("No matching
  #    distribution found for fastmcp") does not mention the interpreter.
  python3.12 -m venv rag-workspace/.poc-venv

  # 3. Dependencies from PyPI (~124 packages).
  rag-workspace/.poc-venv/bin/pip install -r rag-workspace/requirements.txt

  # 4. Model: fetched from our GitHub release, SHA-256 verified, ~4s.
  #    Expect "Ready from the pre-built release asset (no HuggingFace request)."
  rag-workspace/.poc-venv/bin/python3 rag-workspace/download_model.py

  # 5. Build the index (one-off; minutes on a large repo).
  rag-workspace/.poc-venv/bin/python3 rag-workspace/ingest_codebase.py --full

Then register it as a stdio MCP server, using ABSOLUTE paths:

  {
    "mcpServers": {
      "codebase-rag": {
        "type": "stdio",
        "command": "<abs path>/rag-workspace/.poc-venv/bin/python3",
        "args": ["<abs path>/rag-workspace/mcp_server.py"],
        "env": {}
      }
    }
  }

Finally, copy rag-workspace/AGENTS.md into this project's root instruction file
(CLAUDE.md / GEMINI.md / AGENTS.md), or reference it with @rag-workspace/AGENTS.md
if your assistant supports imports. Without that the tools exist but won't get used.

Verify before reporting success:
  - Step 4 printed "no HuggingFace request".
  - `ls rag-workspace/.models_cache/models--*/` shows refs/ and snapshots/ but
    NO blobs/ directory. A blobs/ directory means it fell back to HuggingFace.
  - rag_status() reports a nonzero chunk count and "Token counting : exact".

If step 3 fails because PyPI is unreachable, stop and report that — do not
attempt a workaround. Everything else is already offline.
```

---

## Fetching the model by hand

If you would rather not run the script, or want to move the model onto an
air-gapped machine, the asset unpacks straight into place:

```bash
curl -L -o model-cache.tar.gz \
  "https://github.com/bornohin/rag-builder/releases/download/models-v1/bge-small-en-v1.5-onnx-q.tar.gz"

# Verify BEFORE unpacking. Do not skip this.
echo "3830f02a8ccbfb88dfe2cb460e2e5c6e2e1ac9e106f60c836e7ecf7b4cb42187  model-cache.tar.gz" | shasum -a 256 -c -

tar -xzf model-cache.tar.gz -C rag-workspace/     # creates rag-workspace/.models_cache/
```

Then `download_model.py` finds it already cached and downloads nothing.

### What it contains

Seven files — the model itself plus the one pointer HuggingFace's loader needs:

```
.models_cache/
├── CACHEDIR.TAG
└── models--qdrant--bge-small-en-v1.5-onnx-q/
    ├── refs/main                     40 bytes: names the revision
    └── snapshots/52398278.../
        ├── model_optimized.onnx      63.4 MB
        ├── tokenizer.json             0.7 MB
        ├── config.json
        ├── tokenizer_config.json
        └── special_tokens_map.json
```

No symlinks and no `blobs/`, so it extracts correctly anywhere, including
Windows. `fastembed` loads through `huggingface_hub.snapshot_download`, which is
why the files have to sit at that path rather than in a flat directory — the
five model files alone will not load.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for fastmcp` | The venv is Python 3.9 — macOS's `/usr/bin/python3`. The error never says so. | Delete the venv and recreate it with `python3.12 -m venv` |
| Setup contacted HuggingFace anyway (a `blobs/` directory appeared) | The asset download failed and it fell back | Check the log for `asset unavailable` or `SHA-256 mismatch`; see the allowlist note above |
| `SHA-256 mismatch` | The asset was rebuilt and the pinned digest is newer than the file a CDN is still serving, or the download truncated | Retry; if it persists, compare against the digest in this file |
| Download dies after a redirect | `release-assets.githubusercontent.com` is blocked while `github.com` is allowed | Allowlist that host, or fetch the tarball elsewhere and copy it over |
| `pip install` cannot reach PyPI | PyPI blocked too | Ask for a wheelhouse build — every dependency as wheels for your platform, installable with `--no-index` |
| Everything installs, but the agent never uses the tools | `AGENTS.md` was not wired into the project-root instruction file | See step 6 above |

## Verifying it is genuinely offline

```bash
# Should succeed with HuggingFace unreachable.
HF_ENDPOINT=https://blocked.invalid \
  rag-workspace/.poc-venv/bin/python3 -c "
import sys; sys.path.insert(0,'rag-workspace')
import mcp_server as m; print(m.rag_status())"
```

Nothing else phones home either: ChromaDB's product telemetry, which is enabled
by default upstream, is turned off in `rag_config.chroma_client()`.
