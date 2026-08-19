# Interview Study — Answer Key

Companion to `interview-study-questions.md`. Read a question, write your own
answer, THEN check here.

**Status of these answers.** The Project sections (A–D) are **verified against the
source** at commit `508610b` — every constant, function name and behaviour below was
read out of `rag_config.py`, `rag_chunker.py`, `ingest_codebase.py` and
`mcp_server.py`, not inferred. Where a number came from a measurement, the
measurement is quoted so you can repeat it. Sections E–H are general knowledge and
carry no project claims.

Two things this buys you in an interview: you can state a constant and say *why it
has that value*, and when someone pushes one level deeper you have the mechanism
rather than a paraphrase. If you change the code, re-verify — a stale answer key is
worse than none, because you'll say it with confidence.

---

## Section A — Project: RAG Fundamentals

**1. End-to-end flow (ingest → query)**
*Ingest* (`ingest_codebase.py`): walk the repo root, skipping `EXCLUDE_DIRS` and
lock/generated files → read each file and SHA-1 it → compare against the previous
manifest; unchanged files reuse their existing chunks verbatim → changed files go to
`rag_chunker.extract_chunks()`, which parses with tree-sitter and emits chunks on
symbol boundaries → new chunks are embedded in batches of 256 through the ONNX model
and upserted into ChromaDB with their metadata → progress is checkpointed every 20
batches → finally BM25 is rebuilt over *all* chunks and the whole state (chunks,
manifest, BM25, meta) is written to `rag_index.pkl` via tmp-file + `os.replace`, so
the server never sees a half-written file.

*Query* (`mcp_server.py:search_codebase`): the query gets the model's asymmetric
prefix (`"Represent this sentence for searching relevant passages: "` for bge) and is
embedded → ChromaDB returns the top-N by vector similarity, with category and path
filters pushed down into the query → in parallel BM25 scores the same query tokens
over the whole corpus → both ranked lists go into weighted RRF, plus a symbol boost →
optionally a cross-encoder rerank → each hit is formatted with
`file:start_line-end_line`, the symbol name, and a skeleton preview.

**2. Tree-sitter vs. fixed-size chunking**
Tree-sitter produces a concrete syntax tree, so boundaries land on real units —
`function_declaration`, `class_declaration`, and so on per language in
`CHUNK_TYPES`. Fixed windows cut a function in half, separate a signature from its
body, or straddle two unrelated functions. That damages retrieval twice: the
embedding becomes an average of two half-thoughts, and BM25's term co-occurrence gets
scrambled.

The payoff that matters most in practice is **citation precision**. Because a chunk
*is* a node, `start_line`/`end_line` describe the symbol itself, so a hit can be
handed straight to a file reader. A window index can only ever say "somewhere around
here."

**3. What is a "chunk," and the edge cases**
The unit is whatever the language calls a declaration. Three edge cases, all handled
explicitly in `rag_chunker.py`:

- **Oversized symbol** — if a node exceeds the token budget, `_chunkable_children()`
  first tries to decompose it into its real inner symbols (this is what lets a
  1,200-line `const Provider = () => {…}` become its actual methods). Only if there
  are no structural children does `_split_oversized()` cut it into labelled
  `part i/n` pieces, carrying ~12% overlap so a split symbol stays findable from
  either side.
- **Code no node claimed** — imports, top-level statements, parse errors. `_gap_chunks()`
  window-indexes those with `GAP_WINDOW = 40` lines and `GAP_OVERLAP = 8`, skipping
  gaps under `MIN_GAP_LINES = 4`. This guarantees every line lands in some chunk, so
  recall is never worse than a naive window index.
- **No symbol name resolvable** — `_fallback_symbol()` runs a set of
  declaration regexes covering Go receivers, Rust `pub async unsafe fn`, Kotlin/Dart
  annotations, Swift/Java modifiers and C/C++ definitions. Before that existed, gap
  chunks in those languages came back anonymous and were unreachable by symbol name.

**4. ONNX vs. PyTorch/HuggingFace directly**
Via `fastembed`, the model runs as an ONNX graph on ONNX Runtime — no PyTorch, no
CUDA, no `transformers` at runtime. Concretely: the dependency list is four packages
instead of a multi-gigabyte ML stack, cold start is fast enough that the server can
be a per-session stdio subprocess rather than a long-lived service, and there is no
GPU/driver setup to get wrong on a colleague's laptop.

**5. Why `bge-small-en-v1.5`**
384 dimensions, a 512-token window, strong retrieval quality per FLOP, and it runs on
CPU fast enough that a search is ~10 ms warm. It's also trained with an **asymmetric
instruction prefix**: queries get `"Represent this sentence for searching relevant
passages: "` and passages get nothing. `rag_config.query_prefix()` centralises that
per model family (bge / e5 / arctic / gte) — skipping it measurably degrades recall,
and it's an easy thing to get silently wrong.

Swapping to `bge-base` (768-dim) buys quality at the cost of memory and ingest time.
Any model change is detected in `ingest_codebase.py` (`meta["model"]` mismatch) and
**forces a full rebuild automatically**, because vectors from two models aren't
comparable — see E.9.

**6. ChromaDB vs. BM25 — why separate stores**
They're different data structures for different questions. Chroma holds dense vectors
and does approximate nearest-neighbour search over continuous space; BM25 needs term
statistics — document frequencies, per-document lengths, an average length. No single
structure serves both efficiently, so hybrid systems keep them apart and fuse the
*results*.

Be precise about one thing if pushed: this uses `rank_bm25`'s `BM25Okapi`, which is an
in-memory scorer over tokenised documents, **not** a real inverted index. That's fine
at this scale and is exactly why the scaling answer (D.2) names it as a bottleneck —
`get_scores` does a Python-level dict lookup per document per query term.

**7. BM25 in your own words**
It scores term overlap between query and document, weighted by three things:
term frequency with **saturation** (the 10th occurrence of a word adds far less than
the 2nd, controlled by `k1`), **inverse document frequency** (rare terms are more
discriminative than common ones), and **length normalisation** (controlled by `b`) so
long documents don't win just by containing more words.

It's strong exactly where embeddings are weak: an exact identifier, an error string,
a column name. An embedding model may consider `getUserById` and `fetchUserByID`
similar — useful for prose, but it blurs the sharp exact-match signal BM25 preserves.

**8. "Code-aware" BM25 — the actual implementation**
`rag_config.tokenize()`. Two regexes: `_WORD` pulls out identifiers and numbers,
`_CAMEL` splits camelCase. Every identifier is emitted **whole and decomposed**:

```python
tokenize("getUserToken")  ->  ['getusertoken', 'get', 'user', 'token']
```

So a search for `getUserToken`, `get user token`, or just `token` all hit. Snake_case
is split the same way. There is **no English stemming**, which would corrupt
identifiers.

The critical property is that the *same* function is used at index time and query
time. Asymmetric tokenisation is an invisible bug: nothing errors, results are just
quietly worse. Chunk text is also indexed together with its filepath and symbol name,
so a path fragment matches too.

**9. Line-number tracking through the pipeline**
Tree-sitter gives every node `start_point`/`end_point`, so `_make_chunk()` records
`start_line`/`end_line` at construction. That metadata travels three places: into the
chunk dict in `rag_index.pkl`, into ChromaDB as document metadata, and into the chunk
id itself, which is `"{path}:{symbol}:{start}-{end}"` (plus `.p{n}` for a split part).
No file is re-parsed at query time; formatting a hit is a dict read.

One consequence worth volunteering, because it's a real design cost: **ids encode
line numbers**, so inserting a line near the top of a file changes the id of every
chunk below it, and they all get re-embedded. That's the price of citable ids, and
it's why chunk ids must never be cached across sessions.

**10. Why CPU embeddings**
It serves the portability goal directly — no GPU, no driver setup, runs as a
lightweight per-session subprocess. The cost is throughput: **~15 chunks/second
measured**, so a cold ingest of a 48–64k-chunk corpus is 53–71 minutes. That number
is precisely why ingest checkpointing exists (C.9) — at 133 seconds nobody cares if a
crash restarts; at an hour it ruins an afternoon.

**11. What "100% local/offline" actually buys you**
Technically: no network round-trip in the query path, no third-party uptime or rate
limit, no per-query cost. Practically, and this is the one that matters commercially:
**the source code never leaves the machine**, so it works on an employer or client
codebase where shipping proprietary source to a vendor's embedding API is simply not
allowed. Note the honest boundary — setup does need the network once, to install
packages and cache the model weights. After that, nothing phones home.

**12. How do you know chunks aren't silently truncated?**
This is the best "I measured it" story in the project, so have it ready.

The encoder truncates anything over 512 tokens **silently** — no error, the tail is
just gone, and the chunks that lose their tails are the longest and most informative
ones. The chunker originally budgeted with a constant, `CHARS_PER_TOKEN = 3.0`.
Measuring the real corpus with the encoder's own tokenizer found density ranging from
**1.85 chars/token** (minified, bracket-dense) to ~4.0 (prose), and **11 chunks
(0.6%) were being truncated**.

The fix: count with the encoder's actual tokenizer — `tokenizers.Tokenizer` loaded
from the `tokenizer.json` that fastembed already caches beside the ONNX weights, so no
extra download. The character estimate survives as a cheap pre-filter: anything under
`MAX_CHUNK_TOKENS × 1.6` chars *cannot* overflow whatever it contains, so the exact
count is skipped for the common case and ingest speed is unchanged. `_split_oversized`
also derives its char budget from the measured density of *that specific text* rather
than a global constant, and verifies each resulting piece.

Result: **0 truncated chunks**, and the corpus went from 1,985 to 1,840 chunks —
*fewer*, because prose-dense chunks stopped being needlessly over-split.

---

## Section B — Project: Hybrid Search, Fusion, Ranking

**1. RRF formula**
For a document `d`:

`RRF(d) = Σ over rankers r [ w_r / (k + rank_r(d)) ]`

`rank_r(d)` is d's 1-based position in ranker r's list; `k` (here **60**) is a damper.
`w_r` is the per-ranker weight, which in textbook RRF is 1 for everyone — this system
varies it (B.3).

What `k` actually does: it compresses the gap between top ranks. With k=60, rank 1
scores 1/61 and rank 3 scores 1/63 — close together. Without it (k=0), rank 1 would
be 3× rank 3, and a single confident ranker could hijack every result. The damper is
what makes "appeared reasonably high in *both* lists" beat "topped one list."

**2. Why rank-based fusion instead of summing raw scores**
BM25 scores are unbounded and corpus-dependent; cosine similarity is bounded. Adding
them means whichever happens to produce bigger numbers wins regardless of relevance,
and any normalisation you invent has to be re-tuned as the corpus changes. RRF uses
only rank position, which is directly comparable across any two rankers.

Know the tradeoff too, because it's the obvious follow-up: **RRF throws away
confidence**. A BM25 hit with an enormous score and a marginal one at the same rank
contribute identically. That is precisely the gap the symbol boost (B.5) fills.

**3. "Query-adaptive weighted" RRF — the actual rule**
`rag_config.query_shape()` classifies the query into three shapes, and
`RRF_WEIGHTS` maps each to a `(dense, lexical)` pair:

| shape | rule | dense | lexical |
|---|---|---|---|
| `identifier` | ≤2 words, all bare identifiers, and at least one has an underscore, non-lowercase, or is >12 chars | 0.7 | **1.4** |
| `natural` | ≥4 words with ≥2 stopwords (`how`, `does`, `where`, `the`…) | **1.3** | 0.8 |
| `mixed` | everything else | 1.0 | 1.0 |

The reasoning: a bare `refresh_access_token` that BM25 matched exactly is near-certain
evidence, while a prose question is better served by the dense side. It's deliberately
a cheap, inspectable heuristic rather than a learned classifier — you can read a query
and predict the weights, which matters when debugging a bad result.

**4. The "2 of 8, 0 demoted" evaluation — and its limits**
Concretely: 8 identifier-ish queries, run through both the old (unweighted, unboosted)
fusion and the new one, scoring each by **the rank at which the chunk whose own symbol
matched the query appeared**. Two improved (`sendMessage` 3→1, `presence` 6→1), six
unchanged, none demoted.

Be first to name the weaknesses, because a good interviewer will: 8 queries is a
sanity check, not a measurement; "the chunk whose symbol matches" is a proxy for
relevance, not a human judgment; there are no graded labels; and there's no
statistical power to distinguish a real gain from noise. What would earn trust: a
larger labelled set spanning identifier / prose / multi-hop queries, and a real metric
(nDCG or MRR) rather than a count. Flag it as a known gap — see H.3.

**5. Symbol boost — why it exists on top of RRF**
Because rank cannot express "this chunk **is** the thing you named." A definition and
a file that merely mentions it can arrive at identical ranks from different retrievers,
and RRF has discarded the score that would separate them.

The implementation (`_symbol_boosts`) checks whether a chunk's own `symbol` metadata is
one of the query's tokens. A full match adds `SYMBOL_BOOST / (RRF_K + 1)`, with
`SYMBOL_BOOST = 0.6` — i.e. 60% of one rank-1 RRF step. A partial (camelCase piece)
match adds half that. **Scaling it to a fraction of a rank-1 step is the whole design**:
it's decisive among near-ties and can never override genuine agreement between the two
retrievers.

**6. Identifier query vs. prose query, concretely**
`handle_refresh_token`: `query_shape` → `identifier` → weights (0.7 dense, 1.4
lexical). `tokenize` emits `handle_refresh_token` whole plus `handle`/`refresh`/`token`,
so BM25 hits the exact definition hard. If that chunk's symbol *is* `handle_refresh_token`,
symbol boost adds a further 0.6/61.

`where do we validate expired sessions`: 6 words, stopwords `where`/`do`/`we` → `natural`
→ weights (1.3 dense, 0.8 lexical). BM25 struggles because none of those words need
appear verbatim near the code; the embedding matches code *about* session expiry
regardless of wording. The weighting shifts trust accordingly.

**7. Candidate pool and √(chunks) scaling**
The pool is how many candidates each retriever contributes *before* fusion. RRF can
only rank what retrieval handed it, so too small a pool means the right answer never
gets considered.

The old rule was a fixed multiple, `top_k * 4` = 40 candidates. The problem is that a
fixed *count* is a shrinking *fraction* as the corpus grows: **40 candidates is 2.07%
of a 2,000-chunk index but 0.06% of a 64,000-chunk one.** Same absolute depth,
drastically less coverage.

So the pool now scales with `√(chunks)`, normalised at `POOL_REFERENCE_CHUNKS = 2000`
so small repos get exactly the historical numbers. √ rather than linear because you
want depth to grow with corpus size but far slower — linear would mean 1,280
candidates at 64k, most of them noise, for no recall gain. Measured cost of depth: a
dense query is 9.0 ms at 40 candidates, 10.3 ms at 500, 18.3 ms at 1000, so `POOL_MAX
= 500` is affordable.

**8. What breaks first if the corpus doubles overnight**
Split ingest from query, because they fail differently.

*Query* is fine. BM25 scoring is ~0.95 ms per 1k chunks for an 8-term query, so
doubling 64k→128k takes it from ~60 ms to ~120 ms. ANN search barely moves. The
√-scaled pool absorbs the rest.

*Ingest* feels it first, and the real answer is **resident memory** — ~9.5 MB per 1k
chunks, held for the life of the server because every chunk's text and BM25's
per-document dicts stay in RAM. At 128k chunks that's ~1.2 GB for a background
process. BM25 rebuild is often assumed to be the problem but it's measured at only
~1.6 s at 64k. Memory is the wall; see D.2.

---

## Section C — Project: MCP, Server Design, Systems

**1. What MCP is, plainly**
A standard protocol for exposing tools to an AI assistant: the server advertises tool
names, descriptions and JSON parameter schemas; the client presents those to the model;
when the model chooses one, the client executes it and returns the result. The value is
N×M collapse — any MCP client talks to any MCP server without bespoke glue. This server
exposes 7 tools over plain stdio with no vendor-specific fields, which is why the same
process serves Claude Code, Cursor, Gemini or a custom client unchanged.

**2. stdio vs. HTTP/SSE**
stdio means the server is a subprocess speaking JSON-RPC over stdin/stdout. No port,
no socket, no auth layer, and its lifetime is the session's. That fits "no daemon, no
network" exactly, and it sidesteps authentication entirely because there's no remote
attack surface.

What you give up: one client per process, local machine only, no shared warm cache
across users. For a per-developer code-search tool that's the right trade; for a team
service it wouldn't be.

One practical consequence worth mentioning: **stdout belongs to the protocol.** A
stray `print()` corrupts the JSON-RPC stream and the client just disconnects. That's
why logging goes to `rag.log` via a `log()` helper, and why every tool is wrapped in a
`tool_guard` decorator that converts an exception into a readable string return — an
agent can act on an error message, but a transport fault kills the session.

**3. `pickle.load` and the restricted unpickler**
A pickle is a program for a small stack machine, not a data format. The `REDUCE`
opcode calls a callable with arguments, and `__reduce__` lets an object specify any
callable — `os.system`, `subprocess.Popen`. So unpickling attacker-controlled bytes is
arbitrary code execution with your process's privileges.

`RestrictedUnpickler` subclasses `pickle.Unpickler` and overrides **`find_class`**, the
single hook pickle uses to resolve a module/name pair into a callable. Only 10
allowlisted entries resolve — `rank_bm25`'s BM25 classes, numpy's array reconstructors,
a few collections types, and safe builtins. Anything else raises `UnpicklingError`
*before a single object is constructed*.

Why it's the right fix here: loading a legitimate index is byte-for-byte identical work,
so it costs nothing, while the alternative (migrating off pickle entirely) is a rewrite.
Verified by pickling an object whose `__reduce__` returned `(os.system, ("touch PWNED",))`
— plain `pickle.load` created the file, `load_index_file` refused and it did not.

**4. The seven tools, and when each wins**
- `search_codebase` — hybrid dense+BM25 search, RRF-fused. The default for "where/how
  is X implemented?"
- `find_symbol_references` — where a symbol is **defined** and where it's **used**;
  word-boundary matched, definitions ranked above call sites above mentions, and a
  "zero references" answer is re-checked against a live grep so a stale index can't
  produce a false negative.
- `find_symbol_or_keyword` — literal/regex grep of the working tree, paginated.
- `get_chunk_content` — full verbatim body behind a search hit, by chunk id.
- `get_file_context` — exact line ranges from many files in one call.
- `rag_status` — index health and staleness.
- `reindex` — refresh after edits.

`find_symbol_or_keyword` over `search_codebase` when you need a **guaranteed** answer
rather than a ranked one: proving a string exists or definitively does not (a feature
flag, a column name, a TODO), or right after an edit the index hasn't absorbed. Ranked
search can't prove a negative; grep can.

**5. Why that tool is live grep rather than indexed**
Its entire purpose is being un-stale. The index is a snapshot; a tool that answers
"does this string exist right now" has to hit the filesystem. It prefers `ripgrep` and
falls back to POSIX `grep` so it works on a bare machine, with a configurable timeout
(`RAG_GREP_TIMEOUT`, 30 s) and a message that names which tool ran — because the POSIX
fallback is the one that actually times out on a big repo.

**6. Why batch line-range fetching**
Five regions of interest would otherwise be five tool calls: five round trips, five
model turns, and five chances for one to fail. `get_file_context` takes a list of
`{path, start_line, end_line}` and returns them all at once, which shortens the agent
loop rather than just saving bytes. Since `search_codebase` already returns exact line
ranges, the natural pattern is search once, then fetch every interesting range in a
single follow-up.

**7. What `rag_status` reports, and why it exists**
Repository, indexed file and chunk counts, embedding model, chunker version,
build time, per-repo indexed commits, language breakdown, vector count, whether token
counting is exact, the fusion constants, candidate-pool depth, the path-pushdown limit,
whether a reranker is loaded, and which files have changed on disk since the ingest.

Why it matters: retrieval fails *silently*. A stale index returns plausible, wrong
answers with no error. `rag_status` — plus the `[index staleness]` banner appended to
results — turns a silent failure into a visible one. The design principle is that a
system which can be quietly wrong must be able to report on itself.

**8. `reindex` vs. `--full`**
`reindex()` shells out to `ingest_codebase.py`, adding `--full` only if asked. Default
is incremental: walk, SHA-1 each file, reuse chunks for unchanged files, re-chunk and
re-embed only changed ones, delete vectors for removed files, rebuild BM25 globally,
rewrite the index. `--full` discards the previous state and re-embeds everything. Full
is also *forced automatically* when the embedding model or chunker version changed, or
when Chroma's vector count disagrees with the index — because those mean the existing
vectors can't be trusted.

**9. Content hashing and checkpointing — two mechanisms, don't conflate them**
There are two, and interviewers will probe the difference.

*Incremental manifest.* Each file's SHA-1 is stored with its chunk ids. Why content
hash rather than mtime: mtime changes without content changing on every clone,
checkout or `touch`, causing needless re-embedding — and worse, some tooling doesn't
preserve mtimes at all, which would silently *miss* updates. Line count is weaker
still; two completely different files trivially share one.

*Crash resume.* Embeddings were always durable — Chroma upserts per batch. What was
written only at the very end was the **bookkeeping**. So a crash left a store full of
finished work that the next run couldn't account for, hit the "counts disagree →
rebuild" rule, deleted every vector and started over. The embeddings survived; the
knowledge that they existed didn't. Now the manifest plus the set of embedded ids is
checkpointed every 20 batches.

The subtlety worth volunteering: **resume is gated on the file's SHA-1, not on chunk
ids alone.** Ids encode line numbers, so an edit usually changes them — but an in-place
edit preserving line count would leave ids identical while the content differs, and
resume would skip a chunk that needed re-embedding. Verified: 320 chunks embedded
before a kill, exactly 315 skipped after editing one 5-chunk file.

**10. The multi-repo bug — actual root cause**
`hooks/rag-reindex.sh` resolved its target with `git rev-parse --show-toplevel`. Inside
a git hook that returns **the repo being committed to**, not the workspace root. So a
commit in repo B ran a perfectly correct ingest — scoped to repo B alone. The ingest
rebuilds its manifest from the filesystem walk, so every file in repos A and C was
absent from the walk, therefore classified as removed, therefore their chunks and
vectors were deleted. Caught by a three-repo test: a commit in `beta` cut the index
from 5 chunks across 3 repos to 2 chunks across 1. Fixed by asking `rag_config` for the
workspace root instead.

The generalisable lesson, and it's a good one to have ready: **the bug wasn't in the
deletion logic, it was in the question being asked.** Every individual step was
correct; the input was scoped wrong. Anything that reconciles state against a scan is
only as correct as the scan's boundary — "what's missing from this walk" silently
means "delete it."

**11. Path pushdown and the 400 → 10,000 cap**
Pushdown means the path filter goes into ChromaDB's query as a `$in` clause, so the ANN
search only considers matching files. The alternative — fetch the global top-N then
discard non-matching — starves: if your filter matches 10% of the repo, ~90% of a
40-candidate pool evaporates.

The cap decides when pushdown is used. It was 400, chosen by assumption. Measured:
Chroma resolves 400 values in 8 ms, 10,000 in 13 ms, 32,000 in 34 ms, and only fails
past ~40,000 on SQLite's variable limit. So the cap was ~80× more conservative than the
engine needs, and every filter above it fell back to post-filtering **silently** —
precisely on the repos big enough for pushdown to matter. On a 4,000-file repo any
filter covering more than a tenth of the tree crossed it.

Two lessons: measure the resource before trusting the constant that guards it, and make
degradation visible — the search header now says `POST-filtered` when it happens.

**12. Git hooks vs. `sync_and_index.sh`**
Hooks (`post-merge`, `post-commit`, `post-checkout`, `post-rewrite`) re-index after git
operations, installed into **every** repo in the workspace and chained ahead of any
pre-existing hook rather than overwriting it. `sync_and_index.sh` is the manual path:
fast-forward every repo, skip any with uncommitted work, then index once.

Hooks are "belt-and-braces" because the realistic workflow is pull-everything-then-work,
which the manual script covers, and hooks add latency to every `git pull` for freshness
you rarely need mid-session. Both exist because they fail differently: hooks catch what
you forget, the script catches what hooks miss (uncommitted edits, repos where hooks
weren't installed).

**13. Why must the index artifacts never be committed?**
`rag_index.pkl` and `chroma_db/` contain the **verbatim source text** of every chunk —
the index stores chunk bodies so results can be returned without re-reading files.
Committing them to a public repo publishes the code you indexed, in a form nobody
thinks to review because it looks like a binary blob. They're gitignored, and they
rebuild from scratch in one command, so there is no upside to tracking them. This is a
good answer to have because it shows you think about the data a system *retains*, not
just what it computes.

---

## Section D — Project: Scale, Failure Modes, Tradeoffs

**1–2. Failure order: memory → symbol scan → BM25 → whole-index rewrite**
All four are linear in chunk count; they differ in constant factor, which is what sets
the order. Measured per 1,000 chunks:

| Bottleneck | Cost / 1k chunks | Why it degrades |
|---|---|---|
| Resident memory | **9.5 MB** | Every chunk's text *and* skeleton *and* BM25's per-document frequency dict stay in RAM for the server's life |
| `find_symbol_references` | 3.7 ms | Regex `findall` over **every** chunk's text — no index over symbols exists |
| BM25 scoring | 0.95 ms | `rank_bm25` does a Python-level dict lookup per document per query term |
| Index rewrite | 7 ms | The entire pickle is re-serialised even for a one-file change |

Memory is first because it's a hard wall, not a slowdown: at ~1M chunks it's ~9.5 GB
resident for a background process, and there is no graceful degradation. Symbol scan is
next because 3.7 ms/1k is the largest per-query constant — ~3.7 s at 1M chunks. BM25 is
4× cheaper per unit. The rewrite is last because it's off the query path — but it's the
one that breaks the *promise* of incremental indexing, since a one-line edit pays for
the whole index.

**3. Why BM25 isn't incremental**
Adding one document changes corpus-wide statistics — document frequency for every term
it contains, the average document length, hence every score. Maintaining that
incrementally means updating postings lists and IDF in place, which is what a real
inverted index (Lucene, SQLite FTS5, Tantivy) is built to do. `rank_bm25` isn't an
inverted index at all, so a rebuild is the only correct option. It's measured at ~1.6 s
at 64k chunks, so the simplicity is worth it today — and it's on the list for exactly
when it won't be.

**4. The ×1.28 tiktoken → bge ratio**
Different tokenizers, different vocabularies. tiktoken's `cl100k_base` is a 100k-token
BPE tuned on general web text; bge-small uses BERT-style **WordPiece with a ~30k
vocabulary**, which fragments code harder because it lacks the code-specific merges.
Measured on a real corpus: 225,616 tiktoken tokens → 288,820 bge tokens, **×1.28
overall**, ranging from ×1.04 (CSS) to ×1.53 (SQL) by language.

The practical point: token counts are **not portable**. Capacity planning done in
tiktoken and applied to a BGE model under-counts by ~28%, which silently blows a chunk
budget. Either measure the ratio on representative text or re-tokenize with the target
tokenizer.

**5. Why divide by observed mean tokens/chunk, not a window stride**
A sliding-window estimate is `total_tokens / stride`, which assumes every chunk is the
window size. Syntax-aware chunks are wildly variable — measured **mean 148 tokens,
median 90, against a 440 ceiling**, i.e. only 34% average budget utilisation, because a
three-line property declaration is its own chunk.

So you divide by the *observed* mean. Doing it the naive way for a 6.2M-token corpus
predicted ~14,000 chunks; the real figure is ~58,000 — **4× off**, and every downstream
estimate (memory, ingest time, index size) inherits that error. Worth stating plainly:
the estimate was wrong in the direction that matters, i.e. optimistic.

**6. Why the SQLite migration was declined**
Because the trigger cited for it was ~30k+ files and the actual corpus was 161 files /
1,930 chunks / 2.9 MB. The *security* half of the argument (pickle = code execution)
was real and got fixed at zero runtime cost with the restricted unpickler. The *memory*
half is a genuine rewrite that would also slow scoring.

The explicit revisit trigger: **>50k chunks, or an index past ~200 MB, or `_load_index`
becoming visible in tool latency.** And the right move then isn't "SQLite instead" but a
split — chunk *text* in SQLite read on demand for the ~10 hits actually returned, BM25
postings staying resident. Loading 50k chunk bodies to display 10 is the waste worth
attacking. This is a good judgment answer: it's not "no", it's "not yet, and here's the
number."

**7. Why hand-written extractors for Dart/SQL/Markdown**
Because the generic path measurably produces worse chunks, and each failure is
specific:
- **SQL** — grammars swallow dollar-quoted function bodies, fusing an entire migration
  file into one `statement` node. `_split_sql()` re-splits on top-level DDL keywords so
  each policy/function stays individually addressable.
- **Dart** — declarations lead with the return type, so the generic "first identifier"
  heuristic yields `Future` instead of the method name. `_dart_symbol()` fixes it. Dart's
  grammar also emits signature and body as siblings, so they're rejoined explicitly.
- **Markdown** — its value is heading-bounded sections, not its AST; `_extract_markdown()`
  splits on headings.

"Mis-parses" means wrong symbol names and fused chunks — a regression you'd measure in
the symbol column, not a stylistic preference.

**8. Why the reranker is opt-in**
A cross-encoder scores query and chunk *together* rather than comparing two independent
embeddings, which is where top-3 precision lives. Be accurate about the cost: it's
`fastembed`'s `TextCrossEncoder` running **locally as ONNX**, so it is not an API call
and doesn't break offline operation at query time. What it does require is a **one-off
model download**, which breaks the guarantee that a fresh clone works with no network.

Hence opt-in via `RAG_RERANKER`: unset changes nothing; set-but-uncached logs the reason
and falls back to the fused RRF order rather than failing the search. At the projected
work scale — 48–64k chunks in a crowded 384-dimensional space — it's the first thing
you'd turn on.

**9. "Scale it 10×" — the grounded answer**
Not "add RAM". In order, and each already measured:
1. Move chunk **text** out of the resident index into SQLite keyed by chunk id, read
   on demand for the ~10 hits returned. That's most of the 9.5 MB/1k.
2. Build a `symbol → chunk_ids` map at ingest, turning the 3.7 ms/1k full scan into a
   dict lookup.
3. Replace `rank_bm25` with **SQLite FTS5** — BM25 on disk, near-zero resident memory,
   same dependency, no daemon. This collapses the memory, scoring and rewrite problems
   together.
4. Shard per project (`RAG_INDEX_DIR` + `RAG_COLLECTION` per client entry). Ten 100k
   indexes behave far better than one 1M index, and it matches how people actually
   search a monorepo.

The interview point isn't the list — it's that each item is attached to a measured cost
and a threshold, so you're sequencing work rather than guessing.

**10. A failure mode not yet in the doc**
Answer honestly with something specific. Good candidates: concurrent ingests (a hook and
a manual sync overlapping — the index write is atomic via `os.replace`, but two
processes embedding simultaneously would duplicate work and the second write silently
wins); or **retrieval quality** degradation as opposed to latency — as the corpus grows,
symbol-name collisions across repos become likelier, and nothing currently measures
whether precision@10 is dropping, because there's no eval set big enough to detect it.

That second one is the better answer, because it names the gap the whole of Section H
is about: everything measured so far is *performance*, and almost nothing is *quality*.

**11. A bug where the code asked the wrong question**
Pair this with C.10 — same shape, and having two makes the lesson look like a pattern
you recognise rather than a one-off.

`_detect_repo_root()` called `git rev-parse --show-toplevel` on the workspace directory
to find the repository to index. That worked for as long as the workspace was never
itself a git checkout. When it became one — the workspace is now distributed as a repo
you clone into a project — git correctly answered "the workspace", so the indexer would
have indexed the retrieval tooling instead of the code. And it would have failed
*silently*: `rag-workspace` is in `EXCLUDE_DIRS`, so the result is a near-empty index,
not an error. Fixed by asking about the **parent** directory, which is what it always
meant.

Both bugs share a root: a call that was correct under an unstated assumption, which
stopped holding when the deployment shape changed.

---

## Section E — Fundamentals: Embeddings & Vector Search

**1. What is an embedding**
Mathematically, a fixed-length vector of real numbers that a model produces
from an input (text, image, etc.), trained so that the geometric relationship
between vectors (distance, angle) reflects some notion of semantic similarity
between the inputs. Intuitively: it's a coordinate in a high-dimensional
space where "similar meaning" inputs land near each other.

**2. Cosine similarity — why and when it fails**
Cosine similarity measures the angle between two vectors, ignoring
magnitude — which is desirable when you want "these two texts are about the
same thing" regardless of length/intensity differences that inflate raw
vector magnitude. It's a poor choice when magnitude itself is meaningful
signal (e.g. some embedding schemes encode confidence or intensity in vector
length) — in those cases, Euclidean/L2 distance or a dot product (which
does account for magnitude) may be more appropriate.

**3. Dense vs. sparse retrieval**
Dense (embeddings): represents text as a continuous vector; strong at
semantic/paraphrase matching (different words, same meaning) — e.g. "how do
I cancel my order" matching a doc that says "return or refund a purchase."
Sparse (BM25/TF-IDF): represents text as term-frequency statistics over an
explicit vocabulary; strong at exact/lexical matches — e.g. matching an exact
error code, a product SKU, or a specific function name where word overlap
*is* the signal.

**4. ANN indexes — why approximate**
Exact brute-force nearest-neighbor search requires computing similarity
against every vector in the index — O(n) per query, which becomes
prohibitively slow as n grows into the millions. ANN indexes trade a small,
usually negligible, accuracy loss for large speed gains by using a data
structure that avoids checking every vector — typically achieving sublinear
or near-logarithmic query time at the cost of occasionally missing the
absolute-best match.

**5. HNSW / IVF core idea**
HNSW (Hierarchical Navigable Small World): builds a multi-layer graph where
each vector is a node connected to its approximate near neighbors; search
starts at a sparse top layer and greedily navigates down through
increasingly dense layers, converging on the query's neighborhood in
roughly logarithmic steps rather than scanning everything. IVF (Inverted
File Index): clusters the vector space (e.g. via k-means) into cells;
at query time, only the nearest few cluster cells are searched instead of
the whole index, trading a small recall loss for a large search-space
reduction.

**6. Curse of dimensionality**
As dimensionality increases, distances between points tend to concentrate —
the ratio between the nearest and farthest point shrinks toward 1, meaning
"nearest" becomes statistically less meaningful and harder to distinguish
from "far." This makes exact nearest-neighbor search both computationally
harder and less discriminative in very high dimensions, which is part of
why ANN methods and dimensionality choices in embedding models matter.

**7. Chunking's fundamental tension**
Too-small chunks lose context — a chunk might contain a code fragment or
sentence that's ambiguous or meaningless without its surrounding scope, hurting
both retrieval matching and the usefulness of the retrieved result. Too-large
chunks dilute the embedding — a chunk covering many different concepts
produces a vector that's an average of all of them, matching queries about
any one topic poorly, and also waste context budget when injected into a
prompt. The right size depends on the content's natural unit of meaning
(which is exactly why syntax-aware chunking beats fixed-size for code).

**8. Chunk overlap — why, and when code still needs it**
Overlap (each chunk sharing its last N tokens with the next chunk's start) exists
to stop meaning being lost exactly at a boundary in unstructured prose, where an
idea can span an arbitrary cut point.

The interesting answer for code is that it depends on **why** the boundary is there,
and this system does both:

- **Between whole symbols — no overlap.** The boundary is a real structural edge;
  one function ending and another beginning loses nothing. Overlap here would just
  duplicate content and inflate the index.
- **Where a symbol was *split* because it exceeded the token budget — overlap.**
  That cut is arbitrary, exactly like the prose case, so `_split_oversized` carries
  ~12% of the budget (max 3 lines) into the next piece, so a split symbol stays
  findable from either side.
- **Where the parser claimed nothing** (imports, top-level code, parse errors) —
  overlap, because window boundaries there are arbitrary too: `GAP_OVERLAP = 8`
  lines out of a `GAP_WINDOW = 40`.

Measured, chunk text totals **108% of corpus bytes** — that 8% is the deliberate
seam. Saying "code doesn't need overlap" is the common answer and it's half right;
the precise version is that *structural* boundaries don't need it and *arbitrary*
ones do.

**9. Embedding drift and reindexing**
"Embedding drift" refers to the fact that vectors from different embedding
models (or even different versions of the same model) are not comparable —
they live in differently-shaped, differently-trained vector spaces, so a
vector from model A means nothing when compared via cosine similarity to a
vector from model B. Any time you swap embedding models, every existing
vector in the index is now stale/meaningless relative to new queries embedded
with the new model, which is why a full reindex is mandatory after a model
change — there's no way to "migrate" old vectors forward.

---

## Section F — Fundamentals: LLM Basics

**1. How a transformer decoder generates the next token**
At each step, the model takes the sequence generated so far, computes a
contextualized representation of each token via stacked self-attention +
feedforward layers (attention lets each position weigh relevant earlier
tokens), and produces a probability distribution over the vocabulary for
"what token comes next." A token is sampled (or greedily chosen) from that
distribution, appended to the sequence, and the process repeats — this is
why generation is inherently sequential/autoregressive.

**2. Context window**
The maximum number of tokens (input + output combined, in most
architectures) the model can attend to in a single call. Exceeding it means
either the request is rejected outright, or (depending on the system) the
oldest content is silently truncated/dropped — either way, the model loses
access to information beyond that window, which is exactly why context
management (compaction, summarization, retrieval instead of raw stuffing) is
a real engineering problem, not just a capacity number.

**3. Pretraining vs. fine-tuning vs. in-context learning**
Pretraining: training on a broad corpus to learn general language
patterns/knowledge, producing a base model. Fine-tuning: further training
on a smaller, targeted dataset to specialize behavior (e.g. instruction
following, a specific domain), which updates the model's weights. In-context
learning: no weight updates at all — you provide examples or instructions
directly in the prompt, and the model adapts its behavior for that single
call based purely on what's in the context window.

**4. Temperature**
Temperature scales the probability distribution before sampling the next
token — low temperature sharpens the distribution toward the highest-
probability token(s) (more deterministic, more "confident"/repetitive),
high temperature flattens it (more diverse, more random). Temperature 0
typically means always picking the single highest-probability token
(greedy decoding) — maximally deterministic, same input reliably produces
the same output (modulo any residual system-level nondeterminism).

**5. Tokenization and cross-model token count differences**
Tokenization splits text into subword units from a fixed vocabulary learned
during that model's training (commonly via BPE or similar algorithms).
Different models train different vocabularies on different data, so the same
string segments differently — this is exactly the ×1.28 tiktoken→bge ratio
from Section D.4 — measured, not assumed: a 100k-token BPE vocabulary and a 30k
WordPiece one segment the same code very differently (×1.04 on CSS, ×1.53 on SQL).
Token counts are not portable across tokenizers, and treating them as portable
silently blows whatever budget you planned against.

**6. Why hallucination happens**
The model is a next-token predictor trained to produce plausible-sounding
continuations, not a database with a built-in "I don't actually know this"
signal. When it doesn't have reliable information, it still has to output
*something*, and it will generate the statistically plausible continuation
even when that continuation isn't grounded in fact — there's no separate
internal "confidence check" gating output by default. This is precisely why
external grounding (RAG, tool-verified citations) matters — it gives the
model retrieved facts to condition on, and gives the *system* an independent
way to check the model's claims, rather than trusting the model's self-report.

**7. RAG vs. fine-tuning**
RAG solves "the model needs access to specific/current/private information
it wasn't trained on" — cheaply, updatably (change the index, not the
model), and with traceable provenance (you know which document informed the
answer). Fine-tuning solves "the model needs to behave/respond differently in
style, format, or task specialization" — it changes *how* the model
responds, not what facts it can look up. They're complementary: a fine-tuned
model can still use RAG for facts it wasn't trained on; RAG doesn't fix
formatting/behavior issues fine-tuning would.

**8. System prompt vs. user message**
The system prompt sets the model's operating context/instructions/persona
for the whole conversation and is generally treated with higher priority
than user messages (which is by design — it's meant to be a
harder-to-override instruction layer). The user message is the actual
per-turn input/request. This distinction matters for both harness design
(what belongs in the system prompt — durable behavior/tools — vs. what's a
per-turn user message) and for security (system-prompt-level instructions
should not be trivially overridable by user-supplied text).

---

## Section G — Fundamentals: Agents & Tool Use

**1. What makes something an agent**
An agent is a system where the model's output can trigger actions in the
world (tool calls, code execution, file edits) and the *result* of those
actions feeds back into subsequent model calls, driving further decisions —
as opposed to a single LLM call that just produces text and stops. The
defining feature is the loop: observe → decide → act → observe result →
decide again, continuing until some stopping condition, not just "it uses
tools once."

**2. Tool use / function calling, mechanically**
The model is given a set of tool definitions (name, description, parameter
schema) as part of its context. When it decides a tool is needed, instead of
(or alongside) producing normal text, it emits a structured "tool use" block
specifying which tool and what arguments. The *client/harness* — not the
model — actually executes that tool call against the real system, then
returns the result back to the model as a new message so it can continue
reasoning with that information.

**3. Why tool descriptions matter so much**
The model chooses which tool to call based entirely on the tool's name,
description, and parameter schema — it has no other information about what
the tool actually does. An ambiguous description, or two tools with
overlapping-sounding purposes, causes the model to pick the wrong tool, pass
malformed arguments, or waste a loop iteration on a call that doesn't get
what's needed — directly increasing loop length, cost, and error rate. This
is effectively prompt engineering applied to an API surface.

**4. ReAct-style loop vs. single-shot tool call**
A single-shot tool call is one round: model calls a tool, gets a result,
produces a final answer — done. A ReAct ("Reasoning + Acting") style loop
interleaves explicit reasoning with actions across multiple steps — the
model reasons about what it's learned so far, decides the next action,
observes the result, and repeats, allowing it to course-correct based on
what previous tool calls revealed rather than committing to a single
predetermined plan upfront.

**5. Context rot and mitigations**
As a loop runs many iterations, the accumulated transcript (all prior tool
calls, results, reasoning) grows and can start to dilute the model's
attention on what's actually relevant to the current step, or eventually
exceed the context window outright. Two mitigations: (a) summarization/
compaction — periodically collapse older turns into a concise summary that
preserves key facts while dropping verbose intermediate output; (b) selective
retrieval instead of full history — only pull back specific prior results
when they're needed again, rather than keeping everything resident at all
times.

**6. Agent memory vs. context window**
The context window is what the model can see *in this single call* — it
resets with each new conversation/session by default. "Memory" refers to any
mechanism for persisting information *across* calls or sessions beyond what
naturally fits in one context window — this can be as simple as writing
notes to a file the agent re-reads next time, or a structured external store
(like Claude's own memory filesystem) that's selectively loaded into context
on future turns rather than being part of one continuous window.

**7. Grounding and verifying tool-based claims**
Grounding means an output is directly traceable to actual retrieved/verified
information rather than the model's unaided generation. To verify an agent
didn't hallucinate a citation, the harness can independently check — e.g. did
this exact `file:line` actually appear in a tool result returned earlier in
this session? If the model cites a location that was never actually
retrieved, that's a detectable, mechanical red flag the harness can catch
without needing to trust the model's self-report at all.

**8. Stopping conditions — why more than one**
"Task complete" alone is a soft, model-judged condition that can fail
silently (the model thinks it's done but isn't, or never converges and loops
forever). A robust harness also needs hard external stops: max iteration
count, max token/cost budget, a timeout, and ideally a max-consecutive-
failures count (to catch a stuck loop retrying the same failing action) —
these protect against runaway cost and infinite loops regardless of what the
model itself believes about its progress.

---

## Section H — Fundamentals: Evaluation & Safety

**1. Pure IR metrics**
- **Precision@k**: of the top k results returned, what fraction are actually
  relevant.
- **Recall@k**: of all the relevant documents that exist in the corpus, what
  fraction appear in the top k results.
- **MRR (Mean Reciprocal Rank)**: averages `1/rank` of the first relevant
  result across queries — rewards getting a relevant result near the top,
  specifically.
- **nDCG (normalized Discounted Cumulative Gain)**: accounts for *graded*
  relevance (not just relevant/not) and rewards putting more-relevant results
  higher, with diminishing weight for lower positions.

**2. Evaluating RAG end-to-end**
Beyond pure retrieval metrics, you need to evaluate the *generated answer*
quality: does the final response correctly use the retrieved context,
is it faithful to it (no hallucinated additions), and does it actually
answer the user's question. Common approaches: human-labeled answer quality
scoring, LLM-as-judge scoring against a rubric, or exact-match/F1 scoring
if there's a known correct answer format — plus explicitly checking
faithfulness (does the answer's claims trace back to the retrieved
context) separately from correctness.

**3. Golden set and why 8 queries isn't enough** *(your own gap — own it)*
A golden/labeled set is a fixed collection of representative queries with
known-correct (or human-graded) expected results, used to measure and track
retrieval/answer quality consistently over time. 8 queries is a fine
sanity-check starting point but too small to be statistically meaningful or
to cover the real diversity of query types (identifier lookups, prose
questions, multi-hop questions, edge cases like typos or very short/long
queries) your system will actually face — a real eval set needs enough
breadth and volume that a change's effect isn't just noise.

**4. Offline vs. online evaluation**
Offline evaluation runs against a fixed labeled dataset before deployment —
repeatable, fast, good for catching regressions during development. Online
evaluation measures real usage in production (e.g. click-through/acceptance
of results, user corrections, downstream task success) — captures real query
distribution and real-world edge cases an offline set can't anticipate, but
is slower to get signal from and harder to attribute causally. You need
both: offline to catch regressions quickly and cheaply, online to catch
what offline was blind to.

**5. Regression testing and "0 demoted"**
Regression testing means re-running your existing golden set after every
change and confirming previously-good results didn't get worse, not just
that new cases improved. "0 demoted" specifically protects against the
common trap of local optimization — a change (like the query-adaptive RRF
weighting) that improves the cases you were targeting but silently degrades
some other query type you weren't explicitly testing for. Without checking
for demotions, you could ship a net-negative change while only looking at
the metric you wanted to move.

**6. Risk categories for agentic systems with write access**
- **Destructive/irreversible actions** (deleting or overwriting files) →
  mitigate with dry-run mode, confirmation gates, or sandboxed
  filesystem access.
- **Scope creep / wrong-target actions** (editing the wrong file/repo — like
  your actual multi-repo hook bug) → mitigate with explicit scoping/
  validation before any write.
- **Prompt injection** (malicious content in retrieved/tool data steering the
  model into unintended actions) → mitigate by treating tool outputs as
  untrusted data, not instructions, and constraining what actions can follow
  from data vs. from the actual user's request.
- **Runaway cost/loops** → mitigate with hard iteration/budget caps (Section
  G.8).

**7. "The model said it verified X" ≠ X is verified**
The model's stated confidence or claim of verification is just more
generated text — it's not backed by an actual independent check unless the
harness performed one. The general principle: any claim of correctness that
matters should be checked by a mechanism outside the model's own
self-report — a test that actually runs, a citation that's actually
cross-referenced against real tool output, a diff that's actually applied
and inspected — because the model has no privileged access to ground truth
that the harness itself doesn't also have.

---

## Section I — Harness & Loop

**Deliberately blank.** These answers have to come from your own design, and a
pre-written template here would actively hurt you — this is the section where an
interviewer can tell instantly whether you built the thing or read about it.

Use the questions in the companion doc as a build checklist, and write your answers
here as you go rather than afterwards, while you still remember why you rejected the
alternatives. The detail that lands in an interview is never the final design; it's
the thing you tried first that didn't work.

---

## Section J — "Why" / Behavioral

**1. Defending AI-generated code — suggested framing**
Don't be defensive about it, and don't oversell it either. A grounded
framing: "I used Claude to accelerate implementation, the way I'd use any
modern tool — but I made the architectural decisions, I profiled the
constants that mattered (chunk size limits, candidate pool scaling, the
path-pushdown cap), I found and fixed the multi-repo hook bug, and I can
explain the tradeoff behind every non-obvious design choice in this system."
Then *demonstrate* that by actually answering a technical follow-up
correctly and specifically — that's what actually lands, not the framing
sentence itself. The worst outcome is claiming full authorship of code you
can't explain; the second-worst is being apologetic about using AI tools
well. The right outcome is "I directed this, I understand it, and here's the
proof" — shown, not just stated.

**2. Why a local/offline RAG server as a portfolio project**
Reasonable framing: it's a real, non-trivial systems problem (hybrid
search, rank fusion, scale-aware storage decisions) rather than a thin
wrapper around a hosted API — which demonstrates engineering judgment, not
just API-calling ability. The offline/local constraint specifically forces
real tradeoffs (CPU-only embeddings, no reranker by default, careful memory
planning) that a "just call OpenAI's embeddings API" project never has to
confront — meaning the project actually exercises systems thinking.

**3. Hardest bug/decision — practice this one specifically**
Pick one real one and rehearse it out loud: the multi-repo hook bug is a
strong candidate (clear bug, clear root cause, clear fix, generalizable
lesson about scoping destructive operations). Be ready to describe how you
*actually* found it — what you observed first (chunks from another repo
disappearing), what you checked, what the actual root cause turned out to
be — not just the doc's summary of it.

**4. What you'd do differently starting over**
A strong answer names something concrete and shows hindsight, not vague
self-criticism — e.g. "I'd design the checkpointing and multi-repo scoping
from day one instead of retrofitting it after hitting the bug" or "I'd build
a small formal eval set before doing any ranking tuning, instead of doing the
RRF weighting change first and only having 8 ad hoc queries to check it
against."

**5. A decision you'd now revisit**
Good candidates from your own doc: the reranker being fully opt-in rather
than on-by-default with an offline fallback; or the BM25-rebuilt-globally-
every-time approach, which you already know will need to become incremental
past a certain scale. Naming a real, specific tradeoff you'd reconsider —
with the reasoning for why it was reasonable at the time — reads as far more
credible than claiming the design was perfect.
