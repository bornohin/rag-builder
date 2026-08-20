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
