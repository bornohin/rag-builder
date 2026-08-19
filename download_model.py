#!/usr/bin/env python3
"""Pre-download the local models into the workspace cache.

Run once during setup so the MCP server never needs the network at query time.
The embedding model is whatever rag_config resolves (RAG_EMBED_MODEL to
override). The cross-encoder reranker is only fetched when RAG_RERANKER names
one, which keeps a default install fully offline and small.

    python3 download_model.py
    RAG_RERANKER=Xenova/ms-marco-MiniLM-L-6-v2 python3 download_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_config as cfg
from fastembed import TextEmbedding

os.makedirs(cfg.CACHE_DIR, exist_ok=True)
print("Caching '%s' into %s ..." % (cfg.MODEL_NAME, cfg.CACHE_DIR))
model = TextEmbedding(model_name=cfg.MODEL_NAME, cache_dir=cfg.CACHE_DIR)
vector = list(model.embed(["warm up the onnx session"]))[0]
print("Ready: %d-dimensional embeddings, running fully offline." % len(vector))

# The chunker counts tokens with this file; without it it falls back to a
# conservative char heuristic that over-splits. It ships with the ONNX weights.
print("Tokenizer  : %s" % (cfg._tokenizer_path() or "NOT FOUND (using estimates)"))

if cfg.RERANKER_MODEL:
    print("\nCaching reranker '%s' ..." % cfg.RERANKER_MODEL)
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        encoder = TextCrossEncoder(model_name=cfg.RERANKER_MODEL, cache_dir=cfg.CACHE_DIR)
        list(encoder.rerank("warm up", ["warm up the cross encoder"]))
        print("Reranker ready — searches will now rerank their top "
              "%d candidates." % cfg.RERANK_CANDIDATES)
    except Exception as exc:
        print("Reranker unavailable (%s: %s).\nSearch still works; it just "
              "keeps the fused RRF order." % (type(exc).__name__, exc))
else:
    print("\nReranker   : not configured (optional).\n"
          "  Enable with: RAG_RERANKER=Xenova/ms-marco-MiniLM-L-6-v2 "
          "python3 download_model.py")
