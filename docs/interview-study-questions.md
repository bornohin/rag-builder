# Interview Study Plan — AI Engineering (Advanced) Role

Companion to `interview-study-answers.md`. Try to answer each question yourself —
out loud, in your own words — before checking the answer doc. If you can't
answer a project-specific question from memory, that's your signal to reread
that part of the actual source, not the answer doc.

---

## How to use this

1. Go section by section. Don't skip to the answers.
2. For every "Project" question, answer using **your own words**, then verify
   against the code (`rag_config.py`, `mcp_server.py`, etc.) — not just the
   answer doc. The answer doc is a check, not a substitute for reading the code.
3. For every "Fundamentals" question, write a 3–5 sentence answer from memory.
   These are the questions that get asked regardless of what project you bring.
4. Do one pass for recall, then a second pass a few days later. Spacing beats
   cramming for this kind of material.

Suggested schedule (adjust to your timeline):

- **Days 1–2:** Section A (Project — RAG fundamentals) + Section E (Fundamentals — retrieval/embeddings)
- **Days 3–4:** Section B (Project — hybrid search & ranking) + Section F (Fundamentals — LLM basics)
- **Days 5–6:** Section C (Project — MCP & systems) + Section G (Fundamentals — agents/tool use)
- **Days 7–8:** Section D (Project — scale, failure modes, tradeoffs) + Section H (Fundamentals — evaluation & safety)
- **Day 9:** Section I (harness/loop — once you've built it) + Section J (behavioral/"why")
- **Day 10:** Full run-through, no notes, timed at ~2 min per answer (interview pace)

---

## Section A — Project: RAG Fundamentals

1. Walk me through what happens, end to end, from `ingest_codebase.py --full` to a query returning a result.
2. What is Tree-sitter and why chunk with it instead of fixed-size character or token windows?
3. What's the actual unit of a "chunk" in your system — function? class? something else? What happens to code that doesn't fit neatly into a syntax node (e.g. a huge function, or file-level constants)?
4. Why ONNX for the embedding model instead of just loading a PyTorch/HuggingFace model directly?
5. Why `bge-small-en-v1.5` specifically? What would change if you swapped in `bge-base` or a different embedding model entirely?
6. What does ChromaDB store, and what does BM25 index separately? Why not put both in the same store?
7. Explain BM25 in your own words — what is it actually scoring, and why is it good at what vector search is bad at?
8. What does "code-aware BM25" mean in this project — how is it different from plain BM25 over English text?
9. Every hit returns `file:start_line-end_line`. Walk through how that's tracked from chunking through to the response — where does that metadata live?
10. Why does the system embed on CPU instead of GPU? What's the actual tradeoff you're making?
11. What does "100% local / offline" actually buy you, technically and practically, versus calling an embeddings API?

## Section B — Project: Hybrid Search, Fusion, Ranking

1. What is Reciprocal Rank Fusion (RRF)? Write the formula from memory and explain each term.
2. Why RRF instead of just normalizing and summing the two raw scores (BM25 score + cosine similarity)?
3. What does "query-adaptive weighted" RRF mean here — what signal decides the weighting, and how?
4. You said "2 of 8 test queries improved, 0 demoted" for the RRF change. What does that evaluation actually consist of? What would make you trust that result more?
5. What is "symbol boost" and why does it need to exist on top of RRF at all?
6. Walk through a concrete example: query is `"handle_refresh_token"` (an identifier) versus query is `"where do we validate expired sessions"` (prose). What's different in how each is scored and why?
7. What is the "candidate pool" and why does its size matter for fusion quality? Why scale it with √(chunks) instead of a fixed number or a number linear in corpus size?
8. If you doubled the corpus size overnight, what in the ranking pipeline would you expect to break or degrade first?

## Section C — Project: MCP, Server Design, Systems

1. What is MCP (Model Context Protocol), in plain terms, to someone who's never heard of it?
2. Why stdio transport instead of an HTTP/SSE server? What do you give up and gain?
3. Walk through what happens on a `pickle.load` of an untrusted/attacker-controlled file — what's the actual exploit primitive, and how does a "restricted unpickler" close it?
4. Describe each of your seven MCP tools in one sentence each, and say when an agent would reach for one over another. Specifically: when would an agent use `find_symbol_or_keyword` (live grep) instead of `search_codebase`?
5. Why is `find_symbol_or_keyword` implemented as live grep instead of also going through the index? What's the tradeoff (staleness vs. speed vs. completeness)?
6. What is `get_file_context` for, and why is batching line ranges useful — what would the alternative look like without it?
7. What does `rag_status` report, and why would an agent (or a human debugging) need it mid-session?
8. Walk through what `reindex` does differently from the initial `--full` ingest.
9. Explain the ingest checkpointing mechanism — why SHA-1 of file contents specifically, not a timestamp or a line count?
10. Describe the multi-repo bug that was fixed this session (hooks re-indexing and deleting other repos' chunks) — what was the root cause, and what's the general lesson about state/scope bugs it illustrates?
11. Why is the path-pushdown cap relevant at all — what is "path pushdown" doing, and why was 400 too conservative?
12. What are git hooks doing in `install_hooks.sh`, and why are they described as "optional belt-and-braces" now that `sync_and_index.sh` exists?

## Section D — Project: Scale, Failure Modes, Tradeoffs (this is where seniority shows)

1. Walk through your own deferred-work section as if a staff engineer just asked "what breaks first as this scales, and in what order?"
2. At ~250–300k chunks you note failure order: memory → symbol scan → BM25 → whole-index rewrite. Explain *why* each of those degrades in that specific order — what's the underlying resource or algorithmic complexity behind each one?
3. Why is BM25 rebuilt globally on every ingest instead of incrementally? What would an incremental BM25 update actually require?
4. You project tiktoken tokens → bge tokens at a ×1.28 ratio. Where does that ratio come from, and why can't you just reuse tiktoken's count directly for a non-OpenAI embedding model?
5. Why does chunk count divide by observed mean tokens/chunk rather than by a fixed sliding-window stride? What would using a naive sliding-window estimate have gotten wrong?
6. Why was SQLite migration declined for now, and what specific metric (chunk count / index size) would flip that decision?
7. Why were hand-written extractors kept for Dart/SQL/Markdown instead of unifying everything under Tree-sitter's generic path? What does "mis-parses" mean concretely for those languages?
8. The reranker is opt-in via an environment variable. Why does turning it on conflict with the "100% offline" guarantee, and what does it cost you (latency/dependencies) to enable it?
9. If asked "how would you scale this to 10x the current size," what's your actual answer — not aspirational, but grounded in the bottlenecks you've already identified?
10. What's a failure mode in this system that ISN'T in your handoff doc yet — i.e., what would you go find out empirically if you had another week?

## Section E — Fundamentals: Embeddings & Vector Search

1. What is an embedding, mathematically and intuitively?
2. Why does cosine similarity make sense as a distance metric for embeddings, and when does it not (e.g. when magnitude carries information)?
3. What's the difference between a dense retrieval method (embeddings) and a sparse one (BM25/TF-IDF)? Give a query type each is naturally better at.
4. What is an ANN (approximate nearest neighbor) index, and why do vector DBs use approximate search instead of exact brute-force search at scale?
5. Name two ANN algorithms/index types (e.g. HNSW, IVF) and describe the core idea of at least one.
6. What's the "curse of dimensionality" and how does it relate to nearest-neighbor search quality?
7. What is chunking, and what's the fundamental tension in choosing chunk size (too small vs. too large)?
8. What is chunk overlap, and why might you use it — and why might it be a bad idea for structured content like code?
9. What does "embedding drift" mean, and why does re-indexing become necessary if you change embedding models?

## Section F — Fundamentals: LLM Basics

1. At a high level, how does a transformer decoder generate the next token?
2. What is a context window, and what actually happens when you exceed it?
3. What's the difference between pretraining, fine-tuning, and in-context learning (few-shot prompting)?
4. What is temperature, and what does setting it to 0 actually change about generation?
5. What is tokenization, and why do the same string cost different token counts across models (e.g. GPT vs Claude tokenizers)?
6. What causes hallucination, in mechanistic/plain terms — why can't a model just "know it doesn't know"?
7. What is RAG solving that fine-tuning doesn't, and vice versa? When would you pick one over the other, or both?
8. What is a system prompt versus a user message, and why does the distinction matter for how a model weighs instructions?

## Section G — Fundamentals: Agents & Tool Use

1. What makes something an "agent" as opposed to a single LLM call? Give a precise definition, not a vibe.
2. What is "tool use" / function calling, mechanically — what does the model actually output, and who executes the tool?
3. Why do tool descriptions matter so much — what happens when a tool's description is ambiguous or overlaps with another tool's?
4. What is the difference between a ReAct-style loop and a simpler single-shot tool call?
5. What's context rot / context pollution in a long agent loop, and what are two mitigations?
6. What's the difference between an agent's "memory" and its "context window"? How do systems persist information across turns or sessions?
7. What does "grounding" mean for an agent's output, and how would you verify an agent didn't hallucinate a citation it claims came from a tool call?
8. What's a stopping condition, and why does an agent loop need more than one (e.g. not just "task complete")?

## Section H — Fundamentals: Evaluation & Safety

1. How do you evaluate a retrieval system without an LLM in the loop at all (i.e., pure IR metrics)? Name and define at least two (precision@k, recall@k, MRR, nDCG).
2. How do you evaluate a RAG system end-to-end, where the LLM's answer quality also matters, not just retrieval?
3. What is a "golden set" / labeled eval set, and why is 8 test queries (your current state) a good start but not sufficient?
4. What's the difference between offline evaluation and online (production) evaluation, and why do you need both?
5. What's regression testing in the context of a retrieval or agent system — what does "0 demoted" actually protect against?
6. What are the main categories of risk in an agentic system with file/code write access, and what's one mitigation for each (e.g. sandboxing, dry-run mode, approval gates)?
7. Why is "the model said it verified X" not the same as X being verified? What's the general principle here?

## Section I — Harness & Loop (fill in once built)

1. Draw your agent loop from memory — every step, in order, including where it can exit early.
2. What's your context management strategy across loop iterations — what gets kept, summarized, or dropped?
3. What's your budget/stopping policy — max iterations, max tokens, max cost, or some combination? Why that choice?
4. Walk through one real failure you hit while building the loop, and what you changed in the harness to fix it.
5. How do you verify the loop's output is correct, independent of the model's own claim of success?
6. What would you add next if you had another week on the harness specifically?

## Section J — "Why" / Behavioral (own the AI-generated-code story directly)

1. "This code was written with Claude's help — walk me through how you'd defend that in an interview." (Practice saying this out loud, calmly, with a real answer — see the answer doc for a suggested framing.)
2. Why did you choose a local/offline RAG server as a portfolio project instead of a wrapper around a hosted vector DB?
3. What was the hardest bug or design decision in this project, and how did you actually debug/decide it (not what the doc says — what you personally did)?
4. What would you do differently if you started this project over today?
5. What's a design decision in this project you now think was wrong, or would revisit?
