# PaperWeave: Building a Live Multi-Pipeline RAG Comparison System for Scientific QA

*How we designed, built, and evaluated three competing approaches to question-answering over arxiv papers — with real-time metrics, a live leaderboard, and a graph database you probably haven't used before.*

---

## Why This Exists

Retrieval-Augmented Generation has become the default pattern for building knowledge-grounded LLM applications. But the phrase "RAG" hides enormous implementation variance. A naive dense-retrieval pipeline and a graph-aware traversal system are both "RAG" in name — yet they make fundamentally different trade-offs around latency, token efficiency, hallucination risk, and answer depth.

Most comparisons you read are either toy benchmarks on clean question-answer pairs, or vendor white-papers optimized for their own product. PaperWeave was built to answer a more grounded question: **on a real corpus of scientific papers, with a real user asking a real question right now, which pipeline actually produces the best answer — and can we measure that without waiting for an offline evaluation run?**

The result is a full-stack system with three live pipelines, a live evaluation engine that runs on every query, and a leaderboard updated in real time.

---

## The Three Pipelines

PaperWeave runs three fundamentally different QA strategies side-by-side on the same corpus of arxiv papers.

### Pipeline 1 — LLM-Only (the baseline)

The simplest possible approach: take the user's question, pass it directly to the language model with no retrieval, and return whatever the model produces from its parametric memory. This pipeline has no retrieval step, no vector database, no graph traversal. It is purely a measure of what the model already knows.

Its role in PaperWeave is as the **token baseline**. Every other pipeline's token efficiency is measured as a percentage reduction against the LLM-only token count. A pipeline that retrieves tightly focused context and produces a shorter, more grounded answer scores better on token reduction.

```
Question → LLM → Answer
```

### Pipeline 2 — Basic RAG (vector retrieval with ChromaDB)

The conventional dense-retrieval approach. Papers are chunked with a recursive character splitter (chunk size 1400 tokens, 200-token overlap), embedded with `nomic-embed-text` running locally through Ollama, and stored in a persistent ChromaDB collection called `paperweave_basic_rag`.

At query time, the question is embedded with the same model, and a cosine similarity search returns the top-k chunks. Those chunks are assembled into a numbered context block and injected into a prompt that instructs the model to answer only from the retrieved context and cite sources inline as `[1]`, `[2]`, etc.

```
Question
  → embed(question)
  → ChromaDB.similarity_search(top_k=5)
  → build context string with [1]..[k] numbering
  → LLM("Answer ONLY from the context below...")
  → Answer + citations
```

The prompt is deliberately strict: the model is told not to use outside knowledge, to state what is missing if the context is insufficient, and to use a concise scientific QA tone. This makes hallucination detection meaningful — any claim the model makes that doesn't appear in the retrieved context is flagged.

### Pipeline 3 — TigerGraph GraphRAG (graph-aware retrieval)

This is the most architecturally interesting pipeline. Instead of a flat vector index, the corpus is ingested into TigerGraph, a distributed graph database, where documents are modeled as a graph of entities, relationships, and `DocumentChunk` vertices.

At query time, PaperWeave calls TigerGraph's `answerquestion` endpoint with a hybrid retrieval method, top-k=5, and a configurable number of hops (default: 2). The engine performs:

1. **Vector search** on `DocumentChunk` embeddings to find entry-point vertices
2. **Graph traversal** from those entry points across entity-relation edges for `num_hops` hops
3. **Answer synthesis** where the retrieved subgraph is assembled into a response

The query body looks like this:

```json
{
  "question": "What is attention in transformers?",
  "method": "hybrid",
  "method_params": {
    "top_k": 5,
    "num_hops": 2,
    "indices": ["DocumentChunk"],
    "chunk_only": true,
    "verbose": true
  }
}
```

The key insight is that `DocumentChunk` vectors are the retrieval index — not `Document` or `Content` vertices. This is a non-obvious detail: if you index on `Document` you get coarse matches; if you index on `DocumentChunk` you get precise semantic anchors with graph edges to follow.

One significant engineering challenge we hit: TigerGraph's Entity-Concept-Chunker (ECC) sometimes creates `DocumentChunk` vertices with their `embedding` field empty, which causes the `search` step to return an empty `start_set`, breaking the entire retrieval chain. We wrote a local repair script (`scripts/backfill_graphrag_chunk_embeddings.py`) that detects affected chunks and backfills embeddings directly from the host's Ollama instance, bypassing the graph pipeline entirely.

---

## The Orchestrator: `POST /ask/all`

The most important endpoint in the system runs all three pipelines concurrently and then evaluates the results. The implementation is refreshingly simple:

```python
@router.post("/ask/all", response_model=AskAllResponse)
async def ask_all(payload: AskRequest, request: Request) -> AskAllResponse:
    container = _container(request)
    llm_only, basic_rag, graphrag = await asyncio.gather(
        container.llm_only_pipeline.run(payload),
        container.basic_rag_pipeline.run(payload),
        container.graphrag_pipeline.run(payload),
    )
    responses = {
        "llm-only": llm_only,
        "basic-rag": basic_rag,
        "graphrag": graphrag,
    }
    live_eval = await container.live_evaluation_service.evaluate_live_query(
        question=payload.question,
        responses=responses,
        reference_answer=payload.reference_answer,
    )
    ...
```

`asyncio.gather` runs the three pipelines in parallel. Each pipeline is fully async, so retrieval and LLM calls for all three happen concurrently rather than sequentially. The total wall-clock time is bounded by the slowest pipeline, not their sum.

The result is handed immediately to the live evaluation service, which computes all metrics on the three just-generated answers before the response is returned to the caller.

---

## The Live Evaluation Engine

This is where PaperWeave departs from most RAG comparison systems. There is no "run benchmarks offline and look at a dashboard later" workflow. Every call to `/ask/all` triggers a full evaluation pass on the three answers that were just generated.

### Reference Answer

Evaluation requires a reference to compare against. If the user supplies a `reference_answer` in the request payload, that is used verbatim. Otherwise the system synthesizes a **cross-pipeline consensus reference** by concatenating the three pipeline answers:

```python
def _consensus_reference(self, responses):
    lines = []
    for pipeline in ("llm-only", "basic-rag", "graphrag"):
        response = responses.get(pipeline)
        if response:
            lines.append(f"{pipeline}: {response.answer}")
    return "\n\n".join(lines)
```

This is a pragmatic choice: in live use, you rarely have a gold-standard reference answer ready. Using the combined output of all three pipelines as the reference adds noise (a wrong answer in one pipeline pollutes the reference) but still produces useful relative scores when the pipelines mostly agree.

### BERTScore

BERTScore computes contextual embedding similarity between the candidate answer and the reference. PaperWeave runs both raw F1 and baseline-rescaled F1:

```python
scorer = BERTScorer(model_type=model_type, batch_size=8, lang="en", rescale_with_baseline=False)
scorer._tokenizer.model_max_length = 512  # prevent Rust usize overflow in tokenizers
precision, recall, f1 = scorer.score(candidates, references)
```

The tokenizer cap at 512 is a workaround for a bug where certain BERT variants set `model_max_length` to 2 billion, which overflows an internal Rust integer in the HuggingFace tokenizers library. The bonus pass thresholds are raw F1 ≥ 0.88 and rescaled F1 ≥ 0.55.

### LLM-as-Judge

An independent LLM (default: `gemini-2.5-flash`) evaluates each answer against the reference using a structured prompt that requests JSON output:

```json
{
  "score": 1-5,
  "pass": true/false,
  "factual_correctness": 1-5,
  "grounding": 1-5,
  "completeness": 1-5,
  "hallucination_level": 1-5,
  "scientific_accuracy": 1-5,
  "reasoning": "brief reason"
}
```

The pass threshold is strict: `score >= 4`, `hallucination_level <= 2`, and the `pass` field must be `true`. Using a different model family for the judge than for the pipelines reduces self-evaluation bias.

### Token Metrics

Token reduction is computed relative to the LLM-only baseline for the same question:

```python
token_reduction_pct = ((baseline_tokens - total_tokens) / baseline_tokens) * 100.0
```

Additional token metrics include `answer_token_efficiency` (answer word count / total tokens, measuring how much of the token budget ends up in the actual output), `retrieval_compression_efficiency` (output tokens / context tokens, measuring how well the model distills context), and `prompt_context_share` (context tokens / prompt tokens).

### Hallucination Detection

PaperWeave uses heuristic hallucination signals rather than a separate model:

- **Fabricated citations**: extract `[n]` citation markers from the answer, check that n ≤ number of retrieved chunks. Any citation index out of range is fabricated.
- **Answer-context mismatch**: compute content-term overlap between the answer and the retrieved context. High mismatch (`> 35%`) signals the model is generating from parametric memory rather than the context.
- **Unsupported claim estimate**: split the answer into sentences, count the fraction where lexical overlap with the context is below 0.15. This is a rough proxy for how many sentences have no grounding in the retrieved material.

### Retrieval Quality

Retrieval quality metrics measure how well each pipeline's retrieved chunks actually serve the question:

- **Source overlap**: fraction of expected sources (from the eval dataset) that appear in retrieved sources
- **Citation correctness**: fraction of inline citations (`[1]`, `[2]`, etc.) that point to a valid retrieved chunk index
- **Context relevance**: lexical overlap between the question and the concatenated retrieved context
- **Useful chunk ratio**: fraction of retrieved chunks that have any lexical overlap with the question
- **Duplicate chunk ratio**: fraction of chunks that are identical after normalization

---

## The Hackathon Scoring Formula

The leaderboard score is a weighted combination of four sub-scores, all normalized to [0, 100]:

```
Weighted Score =
    Token Reduction × 30%  +
    Answer Accuracy × 30%  +
    Latency         × 20%  +
    Storytelling    × 20%
```

Each sub-score is computed as:

| Component | Formula |
|---|---|
| **Token Reduction** | `1 - (avg_total_tokens / max_tokens_across_pipelines)` |
| **Answer Accuracy** | `max(bertscore_rescaled_f1, bertscore_raw_f1, judge_score / 5.0)` |
| **Latency** | `1 - (avg_total_latency_ms / max_latency_across_pipelines)` |
| **Storytelling** | `mean(retrieval_hit_rate, citation_correctness, 1 - fabricated_citation_rate, 1 - duplicate_chunk_ratio)` |

All scores are clamped to [0, 1] before weighting. The "storytelling" component is named for its hackathon context: it rewards pipelines that retrieve correctly, cite accurately, and avoid fabricating information — the engineering qualities that make an answer trustworthy.

---

## The Storage Layer

### ChromaDB

The Basic RAG index is a persistent ChromaDB collection stored at `data/chroma`. The collection name is `paperweave_basic_rag`. The embedding function is `OllamaEmbeddings` backed by `nomic-embed-text`, which produces 768-dimensional vectors. PaperWeave wraps ChromaDB with a thin `BasicRAGStore` class that handles corpus status checking, graceful degradation when the index is empty, and diagnostic messages that tell the user exactly what to run to fix the state.

### TigerGraph

TigerGraph runs in Docker alongside the application. The graph name is `PaperWeave`. Ingestion goes through `scripts/ingest_graphrag.py`, which supports PDF and Markdown modes. The GraphRAG pipeline communicates with TigerGraph's REST API using basic authentication.

### Benchmark and Metrics Storage

Benchmark results and live evaluation events are serialized to JSON files. The metrics service appends events to `data/metadata/runtime_metrics.json`. The `GET /evaluation/results` endpoint merges the latest live evaluation event (from the metrics JSON) with any offline benchmark data, prioritizing live over offline when both exist for the same question.

---

## The Frontend

The frontend is Next.js 14. Four main components:

- **`query-form.tsx`**: Submits a question and optional reference answer to `POST /ask/all`. Shows a three-phase loading state: running → evaluating → done. The phase transitions are driven by the real response timeline, not a fake progress bar.
- **`result-card.tsx`**: Displays one pipeline's answer with its sources, token counts, latency breakdown, and live evaluation scores (BERTScore, judge score, hallucination flags).
- **`evaluation-dashboard.tsx`**: Polls `GET /evaluation/results` every 12 seconds to show the live leaderboard. Each row in the leaderboard carries a "live" or "offline" badge depending on whether the metrics came from the most recent live query or from a stored benchmark run.
- **`comparison-chart.tsx`**: Renders a radar chart comparing the four hackathon scoring dimensions across pipelines.

---

## Configuration and Providers

The entire system is configured from a single YAML file (`configs/base.yaml`) with environment variable overrides. This means you can run against local Ollama models for development and swap to a cloud provider (Anthropic, OpenAI, Gemini) by setting two environment variables, with no code changes.

Key config dimensions:

```yaml
providers:
  llm:
    provider: ollama        # or openai, anthropic, gemini
    model: qwen2.5:7b
  embeddings:
    provider: ollama
    model: nomic-embed-text
    dimensions: 768
  judge:
    provider: gemini
    model: gemini-2.5-flash

chunking:
  strategy: semantic
  chunk_size: 1400
  overlap: 200

retrieval:
  basic_rag_top_k: 5
  graphrag_num_hops: 2
```

The judge uses a different provider from the answer pipelines. This is important: using the same model to evaluate its own outputs produces inflated scores.

---

## Getting the Corpus In

PaperWeave ships with scripts for the full ingestion lifecycle:

1. **Download papers**: `scripts/download_arxiv_papers.py` fetches PDFs from arxiv based on a configurable query and category filter.
2. **Parse PDFs**: `scripts/parse_pdfs_builtin.py` or `scripts/parse_with_opendataloader.py` extract text from PDFs.
3. **Build Basic RAG index**: `scripts/build_basic_rag.py` chunks the parsed text and populates ChromaDB. The `--bootstrap-arxiv` flag downloads a small public corpus automatically if you have no local papers yet.
4. **Ingest into GraphRAG**: `scripts/ingest_graphrag.py --mode pdf` ingests PDFs into TigerGraph. After ingest, run `make graphrag-forceupdate` to trigger entity extraction, then run the embedding backfill script if chunks are missing embeddings.

---

## Engineering Decisions Worth Noting

**Why `asyncio.gather` for the pipeline fan-out?** The three pipelines are all I/O-bound — they wait on embedding calls, ChromaDB reads, TigerGraph HTTP requests, and LLM completions. `asyncio.gather` gives us true concurrency without threading overhead. If the LLM-only pipeline finishes in 1.2s, Basic RAG in 2.1s, and GraphRAG in 3.4s, the total wall-clock time is 3.4s, not 6.7s.

**Why a separate judge model?** The pipelines use `qwen2.5:7b` locally. The judge uses `gemini-2.5-flash`. Evaluating your own outputs with the same model tends to produce self-consistent but not necessarily correct scores. A different model family acts as an independent grader.

**Why heuristic hallucination detection instead of a model?** Running a dedicated hallucination detection model on every query would add 1–2 seconds of latency and another external dependency. The heuristic signals (out-of-range citation indices, answer-context term mismatch, unsupported sentence fraction) are fast, deterministic, and provide actionable diagnostics. They are not a substitute for a learned hallucination classifier, but they catch the most common failure modes reliably.

**Why ChromaDB over a managed vector service?** Local persistence with zero external dependencies keeps the development loop fast. ChromaDB's `PersistentClient` writes to a local directory, so `python scripts/build_basic_rag.py` is the entire setup. Production deployment can swap in Pinecone or pgvector by changing the `BasicRAGStore` implementation without touching pipeline code.

**Why the backfill script for TigerGraph embeddings?** The ECC pipeline in TigerGraph's upstream GraphRAG is designed to run entity extraction and embedding as part of a single ingestion job. In some configurations — particularly when using a local Ollama endpoint for the embedding model rather than a cloud API — the embedding job silently fails and leaves `DocumentChunk.embedding` as null. Rather than debugging the upstream pipeline, we built a one-shot repair script that reads all `DocumentChunk` vertices with null embeddings, calls Ollama directly, and writes the embeddings back via TigerGraph's REST API. This separates the "get entities into the graph" problem from the "get vectors into the graph" problem.

---

## What We Learned

**LLM-only is a surprisingly strong baseline for well-studied topics.** For questions about transformer architectures or attention mechanisms, `qwen2.5:7b` produces accurate answers from parametric memory. The RAG pipelines only clearly outperform it on questions about specific papers, recent results, or narrow implementation details that the model's training data doesn't cover well.

**Graph traversal is slower but contextually richer.** GraphRAG's answers on multi-hop questions — "what did paper X say about technique Y and how does that relate to paper Z?" — are noticeably more coherent than Basic RAG's, because the traversal naturally connects related entities. But the latency overhead (TigerGraph query + multi-hop traversal + HTTP round-trip) is real and observable in the metrics.

**Token efficiency and answer quality are often anti-correlated.** A shorter answer produced by a focused retrieval pipeline scores high on token reduction, but if the retrieved context was too narrow, the answer is incomplete and scores low on BERTScore and judge accuracy. The scoring formula's equal weighting of token reduction and answer accuracy reflects this tension.

**Consensus reference answers add noise but remain useful.** When no gold-standard reference is available, using the three pipeline outputs as a consensus reference produces BERTScore numbers that are internally consistent but inflate scores for all pipelines. The evaluation becomes more useful when users supply a reference answer or when running against the offline evaluation dataset, which has curated ground-truth answers.

---

## Running It Yourself

```bash
# clone and set up
git clone <repo>
cd paperweave
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# start Ollama (in a separate terminal)
OLLAMA_HOST=0.0.0.0:11434 ollama serve
ollama pull qwen2.5:7b && ollama pull nomic-embed-text

# bootstrap a small paper corpus and build the RAG index
python scripts/build_basic_rag.py --bootstrap-arxiv

# start the backend
uvicorn app.main:app --reload --port 8008

# start the frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:3000`, type a question about machine learning, and watch all three pipelines answer it simultaneously with live evaluation scores.

For TigerGraph GraphRAG:
```bash
make graphrag-build && make graphrag-up
# create graph "PaperWeave" in the TigerGraph UI
python scripts/ingest_graphrag.py --mode pdf
python scripts/backfill_graphrag_chunk_embeddings.py --batch-size 16
```

---

## Stack Summary

| Component | Technology |
|---|---|
| Backend API | FastAPI (Python 3.13) |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Vector Store | ChromaDB (persistent local) |
| Graph Database | TigerGraph |
| Embeddings | nomic-embed-text via Ollama |
| LLM (pipelines) | qwen2.5:7b / llama3.1:8b via Ollama (or any OpenAI-compatible provider) |
| LLM (judge) | gemini-2.5-flash |
| Semantic similarity | BERTScore (bert_score library) |
| Config | Pydantic + YAML + env overrides |
| Containerization | Docker Compose (optional) |

---

PaperWeave is a working demonstration that you don't need a static benchmark dataset to evaluate RAG pipelines rigorously. Live evaluation on every query — with BERTScore, an LLM judge, token efficiency metrics, and hallucination signals — gives you a meaningful signal every time a user asks a question, and that signal gets richer the more questions you ask.

The code is at [github.com/paperweave](https://github.com) (link TBD). If you have questions about the TigerGraph integration, the evaluation formula, or the consensus reference design, open an issue or reach out directly.
