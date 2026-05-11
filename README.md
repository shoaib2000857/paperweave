# PaperWeave

PaperWeave compares three QA pipelines on the same scientific-paper corpus:

- `LLM-only`
- `Basic RAG`
- `TigerGraph GraphRAG`

It includes:

- `backend/`: FastAPI APIs and pipeline orchestration
- `frontend/`: Next.js comparison UI
- `scripts/`: dataset, parsing, ingestion, indexing, and evaluation utilities
- `configs/`: runtime config
- `graphrag/`: TigerGraph GraphRAG integration
- `docs/`: setup and usage guides

## Recommended Workflow

For active development, run everything manually:

1. start Ollama on the host
2. start the PaperWeave backend
3. start the frontend
4. start the TigerGraph GraphRAG stack only if you want GraphRAG comparisons

The PaperWeave Docker compose file is optional. It is mainly for packaged local runs, not required for day-to-day development.

## Quick Start

1. Create the env file:

```bash
cp .env.example .env
```

2. Create the Python env and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. Start Ollama so both the host app and Dockerized GraphRAG can reach it:

```bash
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

In another terminal:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama list
```

4. Start the backend:

```bash
uvicorn app.main:app --reload --port 8008
```

5. Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

6. If you want TigerGraph GraphRAG, follow:

[docs/setup.md](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/docs/setup.md)

## End-to-End GraphRAG Flow

The working GraphRAG procedure is:

1. `make graphrag-build`
2. `make graphrag-up`
3. create graph `PaperWeave`
4. initialize GraphRAG
5. ingest PDFs or markdown
6. run `forceupdate`
7. backfill chunk embeddings if ECC leaves `DocumentChunk.embedding` empty
8. query GraphRAG with `indices: ["DocumentChunk"]`

Important detail:

- GraphRAG hybrid retrieval works on `DocumentChunk` vectors, not on `Document` or `Content`

## Useful Commands

PaperWeave backend:

```bash
uvicorn app.main:app --reload --port 8008
```

Frontend:

```bash
cd frontend && npm run dev
```

Optional PaperWeave Docker app:

```bash
make docker-build
make docker-up
```

TigerGraph GraphRAG:

```bash
make graphrag-build
make graphrag-up
make graphrag-logs
```

GraphRAG ingest:

```bash
python scripts/ingest_graphrag.py --mode pdf
```

GraphRAG chunk-embedding repair:

```bash
./.venv/bin/python scripts/backfill_graphrag_chunk_embeddings.py --batch-size 16
```

## Current GraphRAG Note

There is an upstream ECC issue in this setup where the GraphRAG pipeline may create `DocumentChunk` vertices and content correctly but still leave `DocumentChunk.embedding` empty. When that happens:

- `search` returns empty `start_set` / `selected_set`
- `answerquestion` fails or returns nothing useful

The local repair script:

[scripts/backfill_graphrag_chunk_embeddings.py](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/scripts/backfill_graphrag_chunk_embeddings.py)

backfills those embeddings directly from the host and restores GraphRAG retrieval.

## API

Main comparison endpoint:

- `POST /ask/all`

Pipeline-specific endpoints:

- `POST /ask/llm-only`
- `POST /ask/basic-rag`
- `POST /ask/graphrag`

See:

[docs/api.md](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/docs/api.md)
