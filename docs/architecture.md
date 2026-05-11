# Architecture

PaperWeave has three answer pipelines that share the same dataset, benchmark set, and telemetry contract.

## Pipelines

1. `LLM-only`
2. `Basic RAG`
3. `TigerGraph GraphRAG`

## Backend Layers

- `core/`: configuration, logging, dependency assembly
- `models/`: shared API and domain schemas
- `services/`: provider adapters and evaluation services
- `pipelines/`: pipeline implementations
- `storage/`: local metadata, metrics, and index persistence
- `api/`: FastAPI routes

## Frontend

The dashboard issues one query and renders all three answers side-by-side with token, latency, cost, retrieval evidence, timing breakdown, and benchmark summaries.
