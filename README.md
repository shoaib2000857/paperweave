# PaperWeave

PaperWeave is a production-oriented benchmark system for comparing `LLM-only`, `Basic RAG`, and `TigerGraph GraphRAG` on the same scientific-paper corpus and the same evaluation questions.

The project is structured around the hackathon requirements:

- `backend/`: FastAPI APIs, pipeline orchestration, metrics, benchmarking, evaluation
- `frontend/`: Next.js dashboard for side-by-side pipeline comparison
- `scripts/`: reproducible dataset, indexing, ingestion, benchmarking, and utility scripts
- `configs/`: YAML and JSON configuration files
- `docker/`: Dockerfiles and compose assets
- `data/`: downloaded papers, parsed corpora, metadata, and benchmark questions
- `evaluations/`: evaluation outputs and reports
- `benchmarks/`: benchmark result artifacts
- `graphrag/`: reserved mount point for the official TigerGraph GraphRAG deployment
- `docs/`: architecture, setup, API, dataset, and benchmarking guides

## Quick Start

1. Copy `.env.example` to `.env`.
2. Review `configs/base.yaml`.
3. Start the backend and frontend with Docker Compose or run them locally.
4. Download the dataset with `scripts/download_arxiv_papers.py`.
5. Build the Basic RAG index, configure TigerGraph GraphRAG, and run benchmarks.

## Key Features

- Configurable LLM and embedding providers
- arXiv dataset builder with resumable downloads and token accounting
- Basic RAG with chunking, embeddings, and FAISS indexing
- Official TigerGraph GraphRAG API integration wrappers
- Benchmarking across tokens, latency, cost, BERTScore, and LLM-as-a-Judge
- FastAPI orchestration backend and Next.js dashboard

See [docs/setup.md](/media/shoaib/STUDYLINUX/Hackathons/TigerGraphRAG/paperweave/docs/setup.md) for the full setup flow.
