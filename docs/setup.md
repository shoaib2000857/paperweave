# Setup Guide

This is the current working procedure for running the full PaperWeave pipeline locally.

It covers:

- PaperWeave backend
- frontend
- Ollama
- TigerGraph GraphRAG
- GraphRAG ingestion
- GraphRAG repair path when chunk embeddings are missing

## 1. Prerequisites

- Python `3.11+`
- Node.js `20+`
- Docker with Compose
- Ollama installed on the host

## 2. Python Environment

From the `paperweave/` root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create the env file:

```bash
cp .env.example .env
```

Install or refresh the local package after dependency changes:

```bash
pip install -e .
```

## 3. Ollama

GraphRAG runs inside Docker, so Ollama must be reachable from both:

- host processes
- Docker containers

Start Ollama like this:

```bash
sudo pkill ollama
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

Verify it:

```bash
curl http://127.0.0.1:11434/api/tags
```

Notes:

- `llama3.1:8b` is the safest fallback completion model
- `qwen2.5:7b` also works if it is actually installed
- `nomic-embed-text` is required for embeddings
- The hackathon judge path uses Hugging Face hosted inference, so set `HF_TOKEN` or `JUDGE_API_KEY` in `.env`

## 4. PaperWeave Backend

Run from the `paperweave/` root:

```bash
uvicorn app.main:app --reload --port 8008
```

## 5. Frontend

```bash
cd frontend
npm install
npm run dev
```

## 6. Optional PaperWeave Docker App

This is optional. If you already run backend and frontend manually, you do not need this.

```bash
make docker-build
make docker-up
```

The PaperWeave compose file is for packaged local runs, not required for normal development.

## 7. Build and Start TigerGraph GraphRAG

From the `paperweave/` root:

```bash
make graphrag-build
make graphrag-up


cd /media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/graphrag/tigergraph-graphrag
docker compose -f docker-compose.paperweave.yml restart graphrag
```

This starts:

- TigerGraph Community
- GraphRAG API on `localhost:8000`
- GraphRAG ECC on `localhost:8001`
- chat-history on `localhost:8002`

Watch logs:

```bash
make graphrag-logs
```

## 8. Verify TigerGraph GraphRAG Health

```bash
curl http://localhost:8000/health
```

Check the graph schema shell:

```bash
docker exec tigergraph /bin/bash -lc 'printf "USE GRAPH PaperWeave\nLS\n" | /home/tigergraph/tigergraph/app/cmd/gsql'
```

## 9. Create the `PaperWeave` Graph

```bash
docker exec tigergraph /bin/bash -lc 'printf "CREATE GRAPH PaperWeave()\n" | /home/tigergraph/tigergraph/app/cmd/gsql'
```

If it already exists, that is fine.

## 10. Initialize GraphRAG

```bash
curl -u tigergraph:tigergraph -X POST http://localhost:8000/PaperWeave/graphrag/initialize
```

This installs:

- GraphRAG schema
- loading jobs
- retrieval queries
- vector queries

## 11. Prepare Data

You can either:

- download papers automatically
- add your own PDFs into `data/raw_pdfs/`

Automatic download:

```bash
python scripts/download_arxiv_papers.py
```

## 12. Ingest into GraphRAG

PDF mode:

```bash
python scripts/ingest_graphrag.py --mode pdf
```

If the dataset is larger and GraphRAG takes a while to preprocess the server-side folder, use a longer timeout:

```bash
python scripts/ingest_graphrag.py --mode pdf --timeout-seconds 1800
```

Markdown mode:

```bash
python scripts/ingest_graphrag.py --mode markdown
```

Text mode:

```bash
python scripts/ingest_graphrag.py --mode text
```

The GraphRAG containers read mounted dataset directories under:

- `/paperweave-data/raw_pdfs`
- `/paperweave-data/parsed_markdown`
- `/paperweave-data/parsed_text`

## 13. Refresh GraphRAG After Ingest

```bash
curl -u tigergraph:tigergraph "http://localhost:8000/PaperWeave/graphrag/forceupdate"
```

This triggers the ECC pass.

## 14. Important Query Detail

For hybrid GraphRAG retrieval, the correct index is:

- `DocumentChunk`

Do not use:

- `Content`
- `Document`
- empty `indices`

for hybrid vector search.

The correct query shape is:

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

## 15. Validate the GraphRAG Graph

Check that documents exist:

```bash
docker exec tigergraph /bin/bash -lc 'printf "USE GRAPH PaperWeave\nRUN QUERY get_vertices_or_remove(\"Document\", \"2605.07847v1\", \"false\", false)\n" | /home/tigergraph/tigergraph/app/cmd/gsql'
```

Check that chunks exist:

```bash
docker exec tigergraph /bin/bash -lc 'printf "USE GRAPH PaperWeave\nRUN QUERY get_vertices_or_remove(\"DocumentChunk\", \"2605.07847v1_chunk_10\", \"true\", false)\n" | /home/tigergraph/tigergraph/app/cmd/gsql'
```

Check one chunk’s text:

```bash
docker exec tigergraph /bin/bash -lc 'printf "USE GRAPH PaperWeave\nRUN QUERY StreamChunkContent(\"2605.07847v1_chunk_10\")\n" | /home/tigergraph/tigergraph/app/cmd/gsql'
```

## 16. Build the Evaluation Dataset

The benchmark expects a JSON array with hand-written reference answers.

You can start from the template:

```bash
python - <<'PY'
from evaluation.dataset import write_dataset_template
write_dataset_template("evaluation/datasets/hackathon_eval.json", count=30)
PY
```

Each record should look like:

```json
{
  "id": "q001",
  "question": "What problem does FlashAttention solve?",
  "ground_truth": "FlashAttention reduces memory traffic in exact attention by using an IO-aware tiled algorithm.",
  "category": "lookup",
  "difficulty": "easy",
  "sources": ["2205.14135v1"]
}
```

Recommended size:

- `30-50` questions
- mix of `lookup`, `comparison`, `multi_hop_reasoning`, and `summarization`

## 17. Run the Hackathon Accuracy Benchmark

The benchmark runs all three pipelines against the same ground-truth set and writes:

- judge pass/fail plus correctness percentage
- BERTScore raw F1
- BERTScore rescaled F1
- token, latency, hallucination, and retrieval metrics

Use the Hugging Face judge path from the guide:

```bash
export HF_TOKEN=your_token_here
python scripts/run_benchmark.py \
  --dataset evaluation/datasets/hackathon_eval.json \
  --judge
```

Outputs land in:

- `evaluation/outputs/benchmark_results.json`
- `evaluation/outputs/judge_results.json`
- `evaluation/outputs/bertscore_results.json`
- `evaluation/outputs/leaderboard.json`
- `evaluation/reports/summary_report.md`

Notes:

- BERTScore now defaults to the Hugging Face `evaluate` backend
- BERTScore is forced to `cpu` by default so benchmark runs do not die on consumer GPU VRAM limits
- the judge defaults to `meta-llama/Llama-3.1-8B-Instruct` through `huggingface_hub`
- if you want a local judge instead, set `JUDGE_PROVIDER=ollama` in `.env`

## 16. Check Chunk Embeddings

This is the key search sanity check:

```bash
docker exec tigergraph /bin/bash -lc 'printf "USE GRAPH PaperWeave\nRUN QUERY vertices_have_embedding(\"DocumentChunk\", false)\n" | /home/tigergraph/tigergraph/app/cmd/gsql'
```

Desired output:

```text
"all_have_embedding": true
"size": 0
```

If `size` is not `0`, GraphRAG search will stay empty because the hybrid query starts from `DocumentChunk.embedding`.

## 17. Repair Missing Chunk Embeddings

In this setup, the upstream ECC pipeline may leave `DocumentChunk.embedding` empty even after ingest and `forceupdate`.

Use the repair script:

```bash
./.venv/bin/python scripts/backfill_graphrag_chunk_embeddings.py --batch-size 16
```

This reads chunk text from TigerGraph, embeds it with local Ollama `nomic-embed-text`, and writes vectors directly onto `DocumentChunk.embedding`.

Recheck:

```bash
docker exec tigergraph /bin/bash -lc 'printf "USE GRAPH PaperWeave\nRUN QUERY vertices_have_embedding(\"DocumentChunk\", false)\n" | /home/tigergraph/tigergraph/app/cmd/gsql'
```

## 18. Test GraphRAG Search

Once chunk embeddings exist:

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

Expected:

- non-empty `start_set`
- non-empty `selected_set`
- non-empty `final_retrieval`

## 19. Test GraphRAG Answer Generation

```bash
curl -s -u tigergraph:tigergraph \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8000/PaperWeave/graphrag/answerquestion \
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

If this fails with Ollama `404`, the configured completion model is missing. Pull it with:

```bash
ollama pull qwen2.5:7b
```

If you prefer a safer fallback, switch GraphRAG completion back to `llama3.1:8b`.

## 20. Use PaperWeave APIs

With backend, frontend, Ollama, and GraphRAG running:

- `POST /ask/llm-only`
- `POST /ask/basic-rag`
- `POST /ask/graphrag`
- `POST /ask/all`

`POST /ask/all` compares all three pipelines side-by-side.

## 21. Common Failures

`/ask/all` or GraphRAG returns `404` from Ollama:

- the model is not installed
- or Ollama is not reachable from Docker

Fix:

```bash
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

GraphRAG `search` returns empty `start_set` and `selected_set`:

- `DocumentChunk.embedding` is missing
- or you used the wrong `indices`

Fix:

- use `indices: ["DocumentChunk"]`
- run `forceupdate`
- if still empty, run the backfill script

`answerquestion` fails but `search` works:

- completion model missing
- or model formatting issue

Fix:

- install the configured model
- or switch completion model to `llama3.1:8b`

`docker exec ... gsql` fails with a lexical error:

- the command was pasted with duplicated `docker exec`

Correct form:

```bash
docker exec tigergraph /bin/bash -lc 'printf "USE GRAPH PaperWeave\nLS\n" | /home/tigergraph/tigergraph/app/cmd/gsql'
```
