# Scripts

Reproducible entrypoints for dataset creation, parsing, indexing, pipeline execution, GraphRAG ingestion, and benchmarking.

## Key Scripts

- `download_arxiv_papers.py`
- `parse_pdfs_builtin.py`
- `parse_with_opendataloader.py`
- `build_basic_rag.py`
- `test_basic_rag.py`
- `run_llm_only.py`
- `run_basic_rag.py`
- `run_graphrag.py`
- `evaluate_answers.py`
- `benchmark_all.py`

For a fresh checkout with no local paper corpus, build Basic RAG with:

```bash
python scripts/build_basic_rag.py --bootstrap-arxiv
```
