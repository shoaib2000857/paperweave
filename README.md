# PaperWeave

PaperWeave is a benchmark and comparison system for three scientific QA pipelines over the same local paper corpus:

- `llm-only`
- `basic-rag`
- `graphrag` using TigerGraph GraphRAG

The project is built to answer one question through all three pipelines, compare cost and latency, and evaluate answer quality with BERTScore and an LLM judge.

## What This Project Does

PaperWeave gives you:

- a FastAPI backend that runs all three pipelines
- a Next.js frontend for side-by-side answer comparison
- a local paper corpus workflow based on arXiv PDFs
- a conventional vector RAG pipeline
- a TigerGraph GraphRAG integration
- benchmark scripts for offline evaluation
- a dedicated frontend evaluation page at `/evaluation`

The core goal is not just token reduction. It is to show whether GraphRAG or Basic RAG is actually better than a plain LLM baseline on a local scientific corpus.

## Current Status

The project is working end-to-end with the following practical notes:

- `llm-only` works locally through Ollama
- `basic-rag` works locally over a Chroma vector index
- `graphrag` works against the TigerGraph GraphRAG API on `localhost:8000`
- GraphRAG startup is fragile on first boot because the API container can race TigerGraph readiness
- GraphRAG hybrid retrieval only works against `DocumentChunk` embeddings
- the upstream ECC pipeline may leave `DocumentChunk.embedding` empty, so a local backfill repair script is included
- the benchmark flow now supports:
  - local Ollama judge
  - Hugging Face judge provider
  - Hugging Face `evaluate` BERTScore backend
  - CPU-forced BERTScore to avoid GPU OOM on consumer cards

## Repository Layout

- `backend/`
  FastAPI application, pipeline orchestration, API routes, evaluation services

- `frontend/`
  Next.js UI for live QA comparison and offline evaluation viewing

- `scripts/`
  dataset download, parsing, indexing, GraphRAG ingestion, embedding repair, benchmark generation and execution

- `configs/`
  runtime YAML config

- `data/`
  local PDFs, parsed text, parsed markdown, JSONL, Chroma store, metadata

- `evaluation/`
  evaluation datasets, benchmark outputs, reports, scoring logic

- `graphrag/tigergraph-graphrag/`
  vendored or locally integrated TigerGraph GraphRAG stack

- `docs/`
  setup and API notes

## Architecture

### 1. LLM-only

This pipeline sends the user question directly to the configured Ollama model with no retrieval.

Use case:

- baseline answer quality
- baseline token cost
- baseline latency

### 2. Basic RAG

This pipeline:

1. builds a local Chroma vector index over parsed paper text
2. retrieves top-k chunks
3. prompts the LLM with only the retrieved context
4. asks for a grounded answer with inline source-style citations

Use case:

- standard retrieval baseline
- lower complexity than GraphRAG
- strong control baseline for hackathon comparison

### 3. TigerGraph GraphRAG

This pipeline:

1. ingests the corpus into TigerGraph
2. creates `Document`, `DocumentChunk`, `Content`, `Entity`, `Community` and related graph structures
3. runs GraphRAG hybrid search through the TigerGraph GraphRAG API
4. returns grounded answers and retrieved graph evidence

Use case:

- multi-hop retrieval
- graph-based reasoning over scientific documents
- comparing structured retrieval against vector-only retrieval

## Prerequisites

- Python `3.11+`
- Node.js `20+`
- Docker with Compose
- Ollama installed on the host

## Quick Start

From the `paperweave/` root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Start Ollama so both the host app and Dockerized GraphRAG can reach it:

```bash
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

In another terminal:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama list
```

Run the backend:

```bash
uvicorn app.main:app --reload --port 8008
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Useful pages:

- `http://localhost:3000/`
  Live multi-pipeline question answering

- `http://localhost:3000/evaluation`
  Benchmark and evaluation dashboard

## Data Flow

### Local Corpus

PaperWeave expects the scientific corpus in one or more of these places:

- `data/raw_pdfs/`
- `data/parsed_text/`
- `data/parsed_markdown/`
- `data/jsonl/`

### Download arXiv papers

To pull a local paper corpus from arXiv:

```bash
python scripts/download_arxiv_papers.py
```

This downloader now:

- prioritizes curated foundational papers first
- aims for about `2M` tokens total
- retries partial PDF downloads
- validates downloaded PDFs
- falls back safely when arXiv rate limits or downloads are incomplete

### Parse or index for Basic RAG

To build the Chroma index:

```bash
python scripts/build_basic_rag.py
```

On a fresh clone, you can bootstrap a small public corpus and build the index in one command:

```bash
python scripts/build_basic_rag.py --bootstrap-arxiv
```

Notes:

- the Basic RAG builder now prefers parsed text / markdown / JSONL over raw PDFs
- malformed PDFs are skipped instead of crashing the build

### Test Basic RAG

```bash
python scripts/test_basic_rag.py "What does the corpus say about retrieval augmented generation?"
```

## TigerGraph GraphRAG Setup

The working GraphRAG path is:

```bash
make graphrag-build
make graphrag-up
```

Then, because the GraphRAG API container often starts before TigerGraph is actually ready, restart the API container once:

```bash
cd graphrag/tigergraph-graphrag
docker compose -f docker-compose.paperweave.yml restart graphrag
```

Why this is needed:

- `depends_on` only waits for container startup
- it does not wait for TigerGraph GSQL readiness
- the first `graphrag` boot can crash with `502 Bad Gateway`
- restarting it after TigerGraph settles usually works immediately

GraphRAG services:

- TigerGraph: `localhost:14240`
- GraphRAG API: `localhost:8000`
- GraphRAG ECC: `localhost:8001`
- chat-history: `localhost:8002`

Watch logs:

```bash
make graphrag-logs
```

Health check:

```bash
curl http://localhost:8000/health
```

Create the `PaperWeave` graph if needed:

```bash
docker exec tigergraph /bin/bash -lc 'printf "CREATE GRAPH PaperWeave()\n" | /home/tigergraph/tigergraph/app/cmd/gsql'
```

Initialize GraphRAG:

```bash
curl -u tigergraph:tigergraph -X POST http://localhost:8000/PaperWeave/graphrag/initialize
```

## GraphRAG Ingestion

PDF mode:

```bash
python scripts/ingest_graphrag.py --mode pdf --timeout-seconds 1800
```

Other modes:

```bash
python scripts/ingest_graphrag.py --mode markdown
python scripts/ingest_graphrag.py --mode text
```

After ingest:

```bash
curl -u tigergraph:tigergraph "http://localhost:8000/PaperWeave/graphrag/forceupdate"
```

## Critical GraphRAG Detail

For hybrid GraphRAG retrieval, the only correct index type in this project is:

- `DocumentChunk`

Do not use:

- `Document`
- `Content`
- empty `indices`

Correct retrieval payload:

```bash
curl -s -u tigergraph:tigergraph \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8000/PaperWeave/graphrag/search \
  -d '{
    "question": "What is the distributional gap between real and simulated user behaviors?",
    "method": "hybrid",
    "method_params": {
      "indices": ["DocumentChunk"],
      "top_k": 5,
      "num_hops": 1,
      "num_seen_min": 1,
      "chunk_only": true,
      "doc_only": false,
      "verbose": true
    }
  }'
```

## GraphRAG Embedding Repair

There is an upstream ECC issue in this setup where:

- `DocumentChunk` vertices are created
- content is loaded correctly
- but `DocumentChunk.embedding` may still be empty

When that happens:

- GraphRAG search returns empty `start_set` / `selected_set`
- GraphRAG answer generation becomes weak or empty

Repair script:

```bash
python scripts/backfill_graphrag_chunk_embeddings.py --batch-size 16
```

This backfills missing chunk embeddings directly from the host and restores GraphRAG hybrid retrieval.

## APIs

Main comparison endpoint:

- `POST /ask/all`

Pipeline-specific endpoints:

- `POST /ask/llm-only`
- `POST /ask/basic-rag`
- `POST /ask/graphrag`

Evaluation endpoints:

- `GET /evaluation/results`
- `GET /evaluation/leaderboard`
- `GET /evaluation/benchmark`
- `GET /evaluation/bertscore`
- `GET /evaluation/judge`
- `GET /evaluation/report`

## Frontend Pages

### `/`

The home page runs all three pipelines on one question and shows:

- answer cards
- token usage
- latency
- source evidence
- live leaderboard
- live BERTScore or BERTSim
- judge correctness percentage
- retrieval and hallucination metrics

### `/evaluation`

The evaluation page now shows:

- active vs offline benchmark view
- dataset summary
- artifact availability
- leaderboard with:
  - weighted score
  - raw BERTScore F1
  - rescaled BERTScore F1
  - judge correctness %
  - judge pass rate
  - latency
- report chart links
- markdown report preview
- per-question answer comparisons

## Evaluation System

PaperWeave now supports the hackathon-style two-method evaluation flow.

### Method 1: LLM-as-a-Judge

The benchmark judge can run through:

- Ollama locally
- Hugging Face hosted inference
- other supported provider backends

Your current `.env` can keep this local with Ollama:

```env
JUDGE_PROVIDER=ollama
JUDGE_MODEL=llama3.1:8b
JUDGE_BASE_URL=http://127.0.0.1:11434
```

### Method 2: BERTScore

BERTScore now supports:

- Hugging Face `evaluate.load("bertscore")`
- local cached scorer usage
- CPU-forced execution by default

Default config:

```env
BERTSCORE_BACKEND=evaluate
BERTSCORE_DEVICE=cpu
```

CPU is the default because the `deberta-xlarge-mnli` scorer can easily OOM consumer GPUs.

## Benchmark Datasets

### Template

You can generate a benchmark template with:

```bash
python - <<'PY'
from evaluation.dataset import write_dataset_template
write_dataset_template("evaluation/datasets/hackathon_eval.json", count=30)
PY
```

### Local arXiv dataset

This repo now includes a local generated benchmark dataset based on your existing parsed arXiv papers:

- [evaluation/datasets/local_arxiv_eval_60.json](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/evaluation/datasets/local_arxiv_eval_60.json)

It was generated from your local `data/parsed_text/*.txt` corpus using:

- [scripts/generate_local_eval_dataset.py](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/scripts/generate_local_eval_dataset.py)

This generator:

- scans local parsed papers
- keeps papers with usable abstracts
- prioritizes important curated papers first
- creates question / ground-truth pairs tied to local arXiv IDs

Generate a new version:

```bash
python scripts/generate_local_eval_dataset.py --count 80 --output evaluation/datasets/local_arxiv_eval_80.json
```

## Run the Benchmark

With the local dataset:

```bash
python scripts/run_benchmark.py \
  --dataset evaluation/datasets/local_arxiv_eval_60.json \
  --judge
```

The benchmark runner now prints detailed progress:

- dataset load
- question number
- per-pipeline run start
- per-pipeline completion with latency, tokens, and source count
- BERTScore phase start/finish
- judge phase start/finish
- final artifact writes

This makes it clear whether the script is still working, slow, or crashed.

## Benchmark Outputs

The benchmark writes:

- `evaluation/outputs/benchmark_results.json`
- `evaluation/outputs/judge_results.json`
- `evaluation/outputs/bertscore_results.json`
- `evaluation/outputs/leaderboard.json`
- `evaluation/outputs/benchmark_records.csv`
- `evaluation/outputs/leaderboard.csv`
- `evaluation/reports/summary_report.md`
- chart PNGs under `evaluation/reports/`

## Important Evaluation Notes

- live evaluation on `/` is not the same as the offline benchmark
- if a real reference answer is supplied, live evaluation uses it
- if no reference answer is supplied, the system may use cross-pipeline similarity for live comparison, but the dedicated benchmark should be treated as the authoritative accuracy path
- the strongest submission-quality dataset is still a hand-curated set of questions and cleaned answers, even though auto-generated local sets are useful bootstraps

## Most Useful Commands

Backend:

```bash
uvicorn app.main:app --reload --port 8008
```

Frontend:

```bash
cd frontend
npm run dev
```

Build Basic RAG:

```bash
python scripts/build_basic_rag.py
```

Bootstrap small public corpus and build Basic RAG:

```bash
python scripts/build_basic_rag.py --bootstrap-arxiv
```

Ingest GraphRAG:

```bash
python scripts/ingest_graphrag.py --mode pdf --timeout-seconds 1800
```

Refresh ECC:

```bash
curl -u tigergraph:tigergraph "http://localhost:8000/PaperWeave/graphrag/forceupdate"
```

Backfill GraphRAG chunk embeddings:

```bash
python scripts/backfill_graphrag_chunk_embeddings.py --batch-size 16
```

Run benchmark:

```bash
python scripts/run_benchmark.py \
  --dataset evaluation/datasets/local_arxiv_eval_60.json \
  --judge
```

## Known Limitations

- GraphRAG API startup still needs manual restart after first boot in many runs
- GraphRAG depends on TigerGraph readiness, not just container presence
- local auto-generated benchmark answers are abstract-based and should be manually polished for final judging
- some PDF text extraction is noisy, especially on newer arXiv papers with complex formatting
- BERTScore on CPU is slower but more reliable than GPU on limited VRAM systems

## Further Reading

- [docs/setup.md](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/docs/setup.md)
- [docs/api.md](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/docs/api.md)
- [scripts/README.md](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/scripts/README.md)
- [evaluation/datasets/local_arxiv_eval_60.json](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/evaluation/datasets/local_arxiv_eval_60.json)
