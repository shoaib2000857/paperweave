# Backend

FastAPI service for running the three benchmark pipelines and reporting metrics.

## Main Responsibilities

- expose `/ask/*`, `/benchmark`, `/metrics`, `/health`
- orchestrate `LLM-only`, `Basic RAG`, and `TigerGraph GraphRAG`
- track tokens, latency, cost, and evaluation metrics
- persist benchmark artifacts to local storage
