# Interview Study — Answer Key

Companion to `interview-study-questions.md`. Read a question, write your own
answer, THEN check here.

**Important caveat on the "Project" sections:** these answers are built from
your handoff doc and general RAG/IR knowledge — they are a strong scaffold,
not a transcript of your actual source. Before an interview, confirm every
Project-section answer against the real code in `rag-workspace/` (function
names, exact parameter names, exact constants). Where I'm inferring rather
than quoting your doc, I've said so.

---

## Section A — Project: RAG Fundamentals

**1. End-to-end flow (ingest → query)**
Ingest: walk the target repo → for each source file, parse with Tree-sitter
to get a syntax tree → split into chunks along syntactic boundaries (functions,
classes, etc.) rather than raw character counts → run each chunk through the
ONNX embedding model to get a vector → store vectors + metadata in ChromaDB,
and simultaneously build/update a BM25 index over the same chunk text →
checkpoint progress (SHA-1 per file) so a crash can resume instead of
restarting. Query: the MCP tool receives a query string → it's run through
both retrieval paths (embedding similarity search in Chroma, BM25 lexical
search) → results from both are combined via query-adaptive weighted RRF →
top results are returned to the caller with exact `file:start_line-end_line`
metadata attached to each chunk.

**2. Tree-sitter vs. fixed-size chunking**
Tree-sitter parses code into a concrete syntax tree, so chunk boundaries can
be placed at meaningful units (function/class boundaries) instead of an
arbitrary character offset. Fixed-size chunking will happily cut a function
in half, split a signature from its body, or straddle two unrelated
functions — which damages both embedding quality (the vector represents a
semantically incoherent blob) and BM25 quality (term co-occurrence gets
scrambled). Syntax-aware chunking keeps each chunk a coherent, self-contained
unit that a retrieval hit can be usefully acted on.

**3. What is a "chunk," and edge cases**
Typically a function, method, or class (the granularity a developer would
naturally reference). Edge cases — a very large function, or file-level
constants/imports that aren't inside any named node — need explicit handling:
either a size-based sub-split within a large node, or a fallback "gap chunk"
for code that doesn't belong to a named symbol. Your handoff doc references
"unnamed gap chunks" for Go/Rust/Kotlin/Swift being fixed via a broadened
fallback symbol regex — confirm exactly how gap chunks are named/handled in
`rag_chunker.py` before the interview, since this is a natural follow-up
question.

**4. ONNX vs. PyTorch/HF directly**
ONNX Runtime is a lighter, framework-independent inference runtime — you
export the trained weights once into the ONNX graph format, then run
inference without needing PyTorch, CUDA, or the HuggingFace `transformers`
runtime as dependencies. This matters directly for two of your stated design
goals: (a) "no daemon, no network at query time" — ONNX has a small,
predictable footprint suited to CPU-only, offline operation, and (b) faster
cold-start/lower memory than loading a full PyTorch model, which matters when
this runs as a stdio subprocess spun up per session rather than a persistent
service.

**5. Why bge-small-en-v1.5**
It's a strong small embedding model for its size class — good retrieval
quality per FLOP, small enough to run comfortably on CPU with low latency
(your doc reports ~10ms searches). Swapping to `bge-base` would likely
improve embedding quality (richer representations) at the cost of slower
inference and more memory — which is exactly the tradeoff your deferred-work
section flags for the 6.2M-token scale target ("consider bge-base"). Any
model swap also invalidates the existing index — embeddings from two
different models aren't comparable, so a full reindex is required (this is
the "embedding drift" fundamental — see Section E.9).

**6. ChromaDB vs. BM25 — why separate stores**
They're fundamentally different data structures solving different problems.
ChromaDB stores dense vectors and does similarity search over continuous
vector space (typically via an ANN index). BM25 needs an inverted index over
tokens/terms — a completely different structure optimized for exact/fuzzy
term matching and term-frequency statistics. There's no single data structure
that's efficient for both; hybrid search architectures universally keep them
separate and fuse results afterward (this is exactly what RRF is for).

**7. BM25 explained**
BM25 scores how well a document matches a query based on term overlap,
weighted by (a) term frequency in the document (saturating — the 10th
occurrence of a word matters much less than the 2nd) and (b) inverse document
frequency (rare terms across the corpus are more discriminative than common
ones), with a length-normalization component so long documents don't win
purely by containing more words. It's excellent at exact/near-exact lexical
matches — e.g. a function name, an error string, a specific identifier —
which is precisely where embeddings can be weak, since an embedding model may
consider `getUserById` and `fetchUserByID` "similar" in a way that blurs the
sharp, exact-match signal BM25 preserves.

**8. "Code-aware" BM25**
Plain BM25 over English text typically tokenizes on whitespace/punctuation
and may apply English stemming — both wrong for code. Code-aware tokenization
needs to handle identifier splitting (`camelCase`, `snake_case`, `kebab-case`
all decomposed into sub-terms so a search for "refresh token" matches
`refreshToken`), preserve meaningful punctuation/operators where relevant,
and avoid English-specific stemming that would corrupt identifiers. Confirm
the exact tokenizer logic in your source before the interview — this is a
likely deep-dive question.

**9. Line-number tracking through the pipeline**
At chunk-creation time (Tree-sitter parse), each chunk's start/end byte or
line offsets in the original file are known directly from the syntax tree
node's position. That start_line/end_line pair is stored as metadata
alongside the chunk in both ChromaDB (as document metadata) and the BM25
index (or a shared side-table), so any retrieval path can attach exact
`file:start_line-end_line` to a hit without re-parsing the file at query
time.

**10. Why CPU embeddings**
Directly serves the "no daemon, no network, no vendor lock-in" goals: a
small model on CPU means zero GPU dependency, so the tool runs on any
developer machine without driver/CUDA setup, and can run as a lightweight
per-session subprocess (stdio MCP) instead of requiring a long-lived GPU
service. The tradeoff is throughput — CPU embedding is slower per-chunk than
GPU, which is why ingest of a large corpus takes real wall-clock time
(your doc: 53–71 min cold ingest at the 6.2M-token scale target) rather than
being near-instant.

**11. What "100% local/offline" actually buys you**
Technically: no network round-trip latency at query time, no dependency on a
third-party API's uptime/rate limits, and the codebase being indexed never
leaves the machine. Practically: works on an air-gapped or corporate network
with no external API allowlisting, no per-query cost, and no risk of
proprietary source code being sent to a vendor's embedding API — which
matters a lot if this tool is meant to be used on client/employer codebases,
not just personal projects.

---

## Section B — Project: Hybrid Search, Fusion, Ranking

**1. RRF formula**
For a document `d`, its RRF score is:

`RRF(d) = Σ over each ranker r [ 1 / (k + rank_r(d)) ]`

where `rank_r(d)` is d's rank position (1st, 2nd, 3rd...) in ranker `r`'s
result list, and `k` is a constant (commonly 60) that dampens the influence
of very high ranks and keeps low-ranked-but-present documents from being
totally negligible. You sum this across your two rankers (vector similarity
rank, BM25 rank) to get a fused score.

**2. Why rank-based fusion instead of raw score summation**
BM25 scores and cosine similarities live on completely different, incomparable
scales (BM25 is unbounded and corpus-dependent; cosine similarity is bounded
[-1,1] or [0,1]). Naively adding raw scores means whichever metric happens to
have larger numbers dominates, regardless of actual relevance. RRF sidesteps
the whole score-normalization problem by using only *rank position*, which is
directly comparable across any two ranking methods.

**3. "Query-adaptive weighted" RRF**
Standard RRF gives each ranker equal weight. Your system adjusts the two
rankers' weights per-query based on some signal about query type — per your
handoff doc, the intuition is "identifier queries trust BM25, prose trusts
vectors." The likely signal is something query-shape-based (e.g. presence of
camelCase/snake_case tokens, punctuation patterns, or fraction of terms that
match known symbol names) that shifts weight toward BM25 for identifier-like
queries and toward vector search for natural-language prose queries. **Confirm
the exact signal/heuristic in `rag_config.py` before the interview** —
this is exactly the kind of "why this constant" question that gets asked.

**4. The "2 of 8, 0 demoted" evaluation**
This reads as a small manually-curated regression test: a fixed set of 8
representative queries with either labeled expected results or before/after
comparison, run against the old (fixed-weight) and new (query-adaptive)
ranking to confirm the change didn't make anything worse ("0 demoted") while
it improved a couple of specific cases. To trust it more, you'd want: a
larger and more diverse query set, explicit relevance labels (not just "looks
better"), and a quantitative metric (e.g. MRR or nDCG shift) rather than an
eyeballed improved/demoted count. This is your most natural bridge into
Section H (formal evaluation) — flag it as a known gap, not a hidden one.

**5. Symbol boost — why it's needed on top of RRF**
Even with query-adaptive weighting, RRF fuses two *general-purpose* rankers.
If the query is an exact known symbol name (e.g. a function that exists
verbatim in the index), you often want to just guarantee that exact
definition surfaces at or near the top — a targeted boost (e.g. exact-match
symbol lookup gets a rank/score bonus) is a cheap, high-precision override on
top of the general fusion, rather than relying on RRF weighting alone to get
it right.

**6. Identifier query vs. prose query example**
`"handle_refresh_token"` — BM25 will score this very highly against a chunk
containing an exact or near-exact identifier match; the query-adaptive
weighting shifts trust toward BM25's ranking, and symbol boost may push an
exact-name match straight to the top. `"where do we validate expired
sessions"` — this is prose with no code-token overlap; BM25 will struggle
(none of these words may appear verbatim near the relevant code) while the
embedding model can capture the semantic intent and match code that talks
about token expiry/session validation even with different wording — so
weighting shifts toward the vector ranker.

**7. Candidate pool size and √(chunks) scaling**
The "candidate pool" is how many top results each individual ranker (BM25,
vector search) contributes into the fusion step before RRF combines them —
too small and you might miss the actually-best result because it wasn't in
either individual top-k; too large and fusion gets slower and noisier.
Scaling with `top_k * 4`, and then further scaling with √(chunks), keeps the
pool proportionate: your handoff doc's own reasoning is that a fixed
percentage-based pool (e.g. "grab 2% of the index") is fine at 2k chunks but
wildly oversized at 64k chunks — √(chunks) grows the pool slower than the
corpus does, keeping candidate-set size sane at scale without needing it to
be either a hard-coded constant or literally proportional to corpus size.

**8. What breaks first if corpus doubles overnight**
Most likely: BM25 rebuild time (rebuilt globally on every ingest — your doc
notes 1.6s at 64k, so this scales, probably close to linearly, with corpus
size) and raw memory footprint (more vectors + more BM25 postings resident).
Query latency itself is more protected because of ANN search and the
√-scaled candidate pool, but the *ingest* pipeline (BM25 rebuild + embedding
generation for new chunks) is the first thing to feel real pain, consistent
with your own stated degradation order (memory → symbol scan → BM25 → full
rewrite) as chunk count climbs.

---

## Section C — Project: MCP, Server Design, Systems

**1. What MCP is, plainly**
Model Context Protocol is a standard way for an AI assistant/agent to
discover and call external tools — like a defined, structured API contract
between "the model" and "the world," so any MCP-compatible client (Claude
Desktop, Claude Code, etc.) can talk to any MCP-compliant server without
custom integration code per tool. Your server exposes 7 tools; any MCP
client can connect to it and use those tools without you writing
client-specific glue code.

**2. stdio vs. HTTP/SSE transport**
stdio means the server is spawned as a local subprocess and communicates
over stdin/stdout — no network socket, no port, no auth layer needed, and it
naturally lives and dies with the client session. This fits "no daemon, no
network" directly. The tradeoff: it's single-client, local-machine-only by
design — you give up the ability to have a persistent server that multiple
remote clients connect to, and you give up things like built-in
authentication/authorization that an HTTP server would need anyway for
remote access.

**3. pickle.load exploit and the restricted unpickler**
Python's `pickle` format isn't just data — deserializing it can invoke
arbitrary object constructors, including a `__reduce__` method that can call
`os.system` or `subprocess` directly. So loading a `.pkl` file from an
untrusted source is equivalent to executing arbitrary code with your
process's privileges. A restricted unpickler works by overriding
`find_class` (the hook pickle uses to resolve which class/function to
reconstruct) to only allow a small, explicit allowlist of safe types (e.g.
basic Python builtins, numpy arrays) — anything else raises instead of
executing. Your doc notes this was "verified against a live `os.system`
payload," meaning you actually tested that a malicious pickle got blocked,
not just reasoned about it.

**4. The seven tools, one sentence each**
- `search_codebase` — hybrid (BM25 + vector) semantic/lexical search over
  indexed chunks, returns ranked results with file:line locations.
- `find_symbol_references` — finds where a given symbol (function/class/etc.)
  is referenced/used across the indexed codebase.
- `find_symbol_or_keyword` — live grep-based lookup, always reflects the
  current on-disk state (never stale relative to the index).
- `get_chunk_content` — fetches the full text of a specific chunk by ID/location.
- `get_file_context` — batch-fetches multiple line ranges (e.g. surrounding
  context) in one call.
- `rag_status` — reports index health/state (chunk counts, staleness, etc.).
- `reindex` — triggers a rebuild/update of the index.

An agent reaches for `find_symbol_or_keyword` over `search_codebase` when it
needs guaranteed-fresh results (e.g. right after a file was just edited and
might not be reindexed yet) or an exact-match lookup rather than a ranked
semantic search.

**5. Why live grep instead of indexed lookup for that tool**
It trades completeness/speed for freshness guarantees. The vector+BM25 index
can be stale relative to disk (until the next `reindex`/sync), so a tool
whose entire purpose is "never wrong about what's on disk right now" has to
bypass the index and hit the filesystem directly. The cost is that grep is
slower and less structured than an indexed lookup at large scale, so it's
used selectively, not as the default search path.

**6. Why batch line-range fetching**
Without batching, an agent context that needs 5 different code regions would
issue 5 separate tool calls — 5 round trips, 5x the overhead, and 5x the
opportunity for one of them to fail. Batching into one `get_file_context`
call reduces tool-call overhead and keeps the agent loop shorter (fewer
iterations to gather the same context), which matters a lot once you're
budgeting loop iterations in the harness.

**7. What rag_status is for**
Likely reports things like chunk/file counts, last ingest time, whether the
index appears stale relative to disk, and possibly per-repo breakdown in the
multi-repo setup. It matters for both a human debugging ("is my index even
up to date?") and potentially for an agent to self-check before trusting
retrieval results in a task where staleness would be costly (e.g. before a
code edit).

**8. reindex vs. initial --full ingest**
`--full` almost certainly does a from-scratch build (clear and rebuild
everything). `reindex` is likely the incremental path — using the SHA-1
checkpointing to only re-process files that changed, add/update/delete their
chunks, and rebuild BM25 (globally, since that part isn't incremental yet)
without redoing embedding work for unchanged files. Confirm this distinction
against the actual CLI flags/functions in `ingest_codebase.py`.

**9. Why SHA-1 for checkpointing, not timestamp/line count**
A file's mtime can change without content changing (touch, checkout, clone),
which would cause unnecessary re-processing — or worse, a tool/CI step might
not preserve mtimes at all, silently causing missed updates. Line count is
even weaker — two completely different file contents can have the same line
count. A content hash (SHA-1) is the only one of the three that's an exact,
tamper-proof proxy for "has this file's content actually changed," which is
what checkpointed resume needs to be correct.

**10. The multi-repo hook bug — root cause and lesson**
Root cause: the hook body was written to always re-index "the repo it fired
in" without properly scoping which repo's chunks should be affected — so
when a hook fired for repo A, the indexing logic ended up deleting chunks
belonging to repos B and C as a side effect (likely because the index-clear
step or a `--full`-style path wasn't scoped by repo). The general lesson:
any operation that clears/rebuilds shared state needs to be explicitly scoped
to exactly what changed — implicit "the current context" scoping is a common
source of destructive bugs whenever the system is used in more than one
context simultaneously. This is a great example to have ready for "tell me
about a bug you fixed" — it has clear cause, clear blast radius, and a clear
generalizable lesson.

**11. Path pushdown and the cap increase**
"Path pushdown" here means filtering candidates by file path *before* (or
as part of) the vector/BM25 search rather than searching everything and
filtering after — this matters when a query is scoped to a subdirectory or
specific repo. The cap (400 → 10,000) was a safety limit on how many path
values could be pushed down as a filter; your doc's reasoning is that this
was measured (empirically) to be far more conservative than ChromaDB
actually needs (32k values in 34ms), and that above the old cap, filtering
was silently falling back to fetching everything and post-filtering in
Python — much slower and easy to not notice was happening.

**12. Git hooks vs. sync_and_index.sh**
Hooks (e.g. post-checkout, post-merge) automatically trigger re-indexing on
git operations, keeping the index fresh without the developer remembering to
run anything. `sync_and_index.sh` is a manual/scheduled daily-workflow script
that pulls all repos and indexes once. They're described as "optional
belt-and-braces" because the daily script alone is sufficient for staying
reasonably fresh — hooks are a nice-to-have for immediacy, not a
correctness requirement, now that there's a reliable manual/scheduled path.

---

## Section D — Project: Scale, Failure Modes, Tradeoffs

**1–2. Failure order: memory → symbol scan → BM25 → whole-index rewrite**
- **Memory** goes first because both the vector index and BM25 postings are
  kept resident, and resident memory grows roughly linearly (or worse, with
  overhead) with chunk count — this is the most basic, least algorithmically
  interesting bottleneck, which is exactly why it hits first.
- **Symbol scan** next — likely a linear or near-linear scan over all known
  symbols for reference-finding; fine at thousands of symbols, increasingly
  slow as the symbol table grows without an index structure over symbols
  themselves.
- **BM25** next — it's rebuilt *globally* on every ingest (not incremental),
  so its rebuild cost grows with total corpus size regardless of how much
  actually changed; at large scale this rebuild time dominates ingest time.
- **Whole-index rewrite** last/worst — if the underlying storage format
  requires a full rewrite for any update (rather than incremental
  insert/update), this is the most expensive operation and the one that
  eventually forces an architecture change (which is exactly why SQLite
  migration is flagged as the fix once you cross that threshold).

**3. Why BM25 isn't incremental today**
An incremental BM25 update would require updating inverted-index postings
lists and corpus-wide statistics (document frequency, average document
length) that every score calculation depends on — adding or removing a
single document changes IDF for every term it contains, in principle. Doing
this correctly incrementally is more complex than full rebuild; most simple
BM25 implementations (including likely yours) just accept O(rebuild) cost
because it's simpler and, at current scale (1.6s at 64k), still cheap enough
not to matter yet.

**4. The ×1.28 tiktoken→bge ratio**
Different tokenizers (OpenAI's tiktoken BPE vocabulary vs. the tokenizer
`bge-small-en-v1.5` actually uses, likely a BERT-style WordPiece vocabulary)
segment the same text into different numbers of tokens because they have
different vocabularies and merge rules. You can't reuse a tiktoken count
directly for capacity planning against a BGE-based model — you have to
either measure the ratio empirically on representative text (which is what
your doc's number implies happened) or re-tokenize with the actual target
tokenizer.

**5. Why divide by observed mean tokens/chunk, not sliding-window stride**
Your chunks are syntax-aware and variable-length (a one-line function vs. a
50-line class), not fixed-size windows — so a sliding-window estimate
(`total_tokens / window_size`) would implicitly assume uniform chunk size,
which is false here. Dividing by the *empirically observed* mean tokens per
chunk correctly accounts for your chunker's actual size distribution. Your
doc notes this produces ~4x more chunks than a naive sliding-window estimate
— meaning your real chunks are meaningfully smaller on average than a
generic window would predict (consistent with syntax-aware chunking often
producing many small chunks, e.g. short functions, rather than uniform-size
blocks).

**6. Why SQLite migration was declined for now**
Premature — the current pickle+Chroma approach works fine below the
identified threshold (~30k+ files triggers the concern per your doc), and
migrating storage backends is real engineering effort with real risk. The
explicit trigger for revisiting it (per your doc): above ~50k chunks or
200MB index size. This is a good example of deliberately *not* over-engineering
— worth stating explicitly in an interview as evidence of judgment, not
just as a fact.

**7. Why hand-written extractors for Dart/SQL/Markdown**
Generic Tree-sitter-based chunking assumes a "normal" programming-language
structure (functions/classes as the natural unit). Markdown has no such
structure (headings/sections aren't "functions"), SQL's meaningful units
(statements, schema objects) don't map cleanly onto a generic code-parsing
heuristic, and Dart apparently has structural quirks the generic path
mishandles. "Mis-parses" concretely means the generic path would produce
chunks that split mid-statement, or fail to identify natural break points —
hand-written extractors encode the actual structure of each of those formats
instead of forcing a code-shaped heuristic onto non-code-shaped content.

**8. Why the reranker is opt-in**
A reranker is typically a second-stage model (often a cross-encoder) that
re-scores the top candidates after initial retrieval for higher precision —
but that requires either loading an additional model (more memory, more
startup time) or, in some implementations, calling an external API (which
would violate offline operation entirely). Making it opt-in via
`RAG_RERANKER` preserves the default "100% offline, no extra dependency"
guarantee while letting a user who wants better precision (and is willing
to pay latency/memory cost) turn it on explicitly.

**9. "How would you scale this 10x" — grounded answer**
Not "add more RAM" — the grounded answer is: migrate storage to SQLite
(FTS5 for BM25 + chunk text on disk instead of resident) before hitting the
~50k chunk / 200MB threshold, since that's the already-identified next
bottleneck; consider `bge-base` for quality if latency budget allows;
evaluate sharding per-project rather than one global index, since your
doc's own alternative to SQLite migration is "shard per project" — this
also naturally bounds symbol-scan cost per shard. The key interview point:
you're not guessing, you already measured where it breaks and already wrote
down the fix.

**10. What's NOT yet in the handoff doc — a "further work" answer**
A strong honest answer here is something like: BM25 rebuild latency under
concurrent ingests (what happens if two `reindex` calls or a hook and a
manual sync overlap?), or retrieval quality degradation specifically (not
just latency/memory) as symbol name collisions become more likely at scale
across many repos. Being ready to name a real, specific unknown — rather
than claiming everything's covered — is itself a strong signal.

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

**8. Chunk overlap — why, and why not for code**
Overlap (e.g. each chunk shares its last N tokens with the next chunk's
start) helps prevent meaning from being lost exactly at a chunk boundary in
unstructured prose, where sentences/ideas can span an arbitrary cut point.
For code with syntax-aware chunking, boundaries are already placed at
meaningful structural edges (function/class boundaries) — adding overlap
there mostly just duplicates content and adds noise/redundancy without the
same boundary-loss problem overlap solves in prose.

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
issue from Section D.4: you cannot assume token counts are portable across
models/tokenizers.

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

**3. Golden set and why 8 queries isn't enough**
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

Fill this in once the loop is built — these answers should come from your
own design, not a template. Use the questions in the companion doc as a
checklist while you build, and write your answers here (or directly from
memory) once it's working. This section deliberately has no pre-written
answers: if you can answer these fluently from your own implementation,
that's the strongest signal in the whole interview.

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
