# Setup Guide

## Local Prerequisites

- Python 3.11+
- Node.js 20+
- Docker and Docker Compose
- Ollama with `qwen2.5:7b` or another local instruct model
- Ollama embedding model `nomic-embed-text`
- Optional cloud provider keys for Gemini or OpenAI

## Local Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8008
```

## Ollama Setup

The default local stack uses Ollama for:

- answer generation: `qwen2.5:7b`
- embeddings: `nomic-embed-text`
- evaluation/judging: `gemma3:12b`

```bash
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

In another terminal:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama pull gemma3:12b
ollama list
```

Verify the API before calling `/ask/all`:

```bash
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","prompt":"ping","stream":false}'
```

If `/api/generate` returns `404`, Ollama is reachable but the configured model is usually missing. Pull the model named by `LLM_MODEL`, or change `LLM_MODEL` to a model shown by `ollama list`.

## Local Frontend

```bash
cd frontend
npm install
npm run dev
```

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

Docker publishes the backend on host port `8009` by default to avoid colliding with local `uvicorn` on `8008`.

Recommended workflow:

```bash
make docker-build
make docker-up
```

After the first build, do not pass `--build` unless dependencies or Dockerfiles changed. Backend and frontend source directories are bind-mounted, so normal source edits do not require rebuilding.

To choose another backend host port:

```bash
BACKEND_HOST_PORT=8010 NEXT_PUBLIC_API_BASE=http://localhost:8010 docker compose -f docker/docker-compose.yml up
```

When running with Docker, keep Ollama running on the host. The compose file maps container LLM URLs to `http://host.docker.internal:11434`.

## TigerGraph GraphRAG

PaperWeave uses the official TigerGraph GraphRAG repo under `graphrag/tigergraph-graphrag` with a PaperWeave-specific config and compose file.

### 1. Build and start GraphRAG

```bash
make graphrag-build
make graphrag-up
```

This starts:

- TigerGraph Community on `localhost:14240`
- TigerGraph GraphRAG API on `localhost:8000`
- GraphRAG ECC on `localhost:8001`
- Chat history service on `localhost:8002`

GraphRAG uses host Ollama through `http://host.docker.internal:11434`, so Ollama must be running before you query GraphRAG.

### 2. Verify service health

```bash
curl http://localhost:8000/health
docker exec tigergraph /bin/bash -lc '/home/tigergraph/tigergraph/app/cmd/gadmin status'
```

### 3. Create the `PaperWeave` graph

```bash
docker exec tigergraph /bin/bash -lc "/home/tigergraph/tigergraph/app/cmd/gsql 'CREATE GRAPH PaperWeave()'"
```

If it already exists, TigerGraph will tell you.

### 4. Initialize GraphRAG on `PaperWeave`

```bash
curl -u tigergraph:tigergraph -X POST http://localhost:8000/PaperWeave/graphrag/initialize
```

This installs the GraphRAG schema, loading jobs, and search queries.

### 5. Ingest data before querying

Do not start by calling `answerquestion` on an empty graph. Ingest documents first.

Download a starter corpus or add a few PDFs manually:

```bash
python scripts/download_arxiv_papers.py
```

Then ingest through the PaperWeave wrapper script:

```bash
python scripts/ingest_graphrag.py --mode pdf
```

For markdown:

```bash
python scripts/ingest_graphrag.py --mode markdown
```

The script uses GraphRAG `server` ingestion mode and the GraphRAG containers read the mounted dataset path at:

```text
/paperweave-data/raw_pdfs
/paperweave-data/parsed_markdown
/paperweave-data/parsed_text
```

### 6. Refresh the graph after ingest

```bash
curl -u tigergraph:tigergraph "http://localhost:8000/PaperWeave/graphrag/forceupdate"
```

### 7. Test retrieval first

Check that GraphRAG can retrieve non-empty evidence before testing answer generation:

```bash
curl -s -u tigergraph:tigergraph \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8000/PaperWeave/graphrag/search \
  -d '{
    "question": "What is GraphRAG?",
    "method": "hybrid",
    "method_params": {
      "indices": [],
      "top_k": 3,
      "num_hops": 1,
      "num_seen_min": 1,
      "chunk_only": true,
      "doc_only": false,
      "verbose": true
    }
  }'
```

If `selected_set` and `start_set` are empty, the graph still has no usable ingested content.

### 8. Test answer generation

Once retrieval is non-empty:

```bash
curl -u tigergraph:tigergraph \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8000/PaperWeave/graphrag/answerquestion \
  -d '{
    "question": "What is GraphRAG?",
    "method": "hybrid",
    "method_params": {
      "indices": [],
      "top_k": 3,
      "num_hops": 1,
      "num_seen_min": 1,
      "chunk_only": true,
      "doc_only": false,
      "verbose": true
    }
  }'
```

### 9. Use PaperWeave APIs

After GraphRAG is working, run the PaperWeave backend and frontend manually:

```bash
uvicorn app.main:app --reload --port 8008
cd frontend && npm run dev
```

`POST /ask/all` runs three pipelines: LLM-only, Basic RAG, and GraphRAG. If the TigerGraph GraphRAG API is not running, the GraphRAG section will report `graph_unavailable`; use `POST /ask/llm-only` or `POST /ask/basic-rag` when you only want the local Ollama-backed paths.

The GraphRAG endpoint is built from:

```bash
GRAPHRAG_API_BASE=http://localhost:8000
TIGERGRAPH_GRAPH_NAME=PaperWeave
```

Expected call:

```text
http://localhost:8000/PaperWeave/graphrag/answerquestion
```
