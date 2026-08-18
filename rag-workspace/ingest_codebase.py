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

    if not full and prev_chunks and collection.count() != len(prev_chunks):
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

    if new_chunk_ids:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=cfg.MODEL_NAME, cache_dir=cfg.CACHE_DIR)
        log("Embedding %d chunks..." % len(new_chunk_ids))
        batch = cfg.EMBED_BATCH_SIZE
        for i in range(0, len(new_chunk_ids), batch):
            window = new_chunk_ids[i:i + batch]
            texts = [chunks[cid]["text"] for cid in window]
            vectors = [v.tolist() for v in model.embed(texts)]
            collection.upsert(
                ids=window,
                embeddings=vectors,
                metadatas=[chunks[cid]["metadata"] for cid in window],
            )
            log("  %d/%d" % (min(i + batch, len(new_chunk_ids)), len(new_chunk_ids)))

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
            "chunk_count": len(chunks),
            "file_count": len(manifest),
            "tree_sitter": chunker.TREE_SITTER_AVAILABLE,
        },
    }
    tmp = cfg.INDEX_PATH + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cfg.INDEX_PATH)      # atomic: the server never sees a half file

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
