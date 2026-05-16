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
- `run_benchmark.py`
- `generate_eval_questions.py`

For a fresh checkout with no local paper corpus, build Basic RAG with:

```bash
python scripts/build_basic_rag.py --bootstrap-arxiv
```

For the hackathon evaluation flow:

```bash
python - <<'PY'
from evaluation.dataset import write_dataset_template
write_dataset_template("evaluation/datasets/hackathon_eval.json", count=30)
PY

python scripts/run_benchmark.py \
  --dataset evaluation/datasets/hackathon_eval.json \
  --judge
```

This runs:

- `llm-only`
- `basic-rag`
- `graphrag`

and writes:

- `evaluation/outputs/benchmark_results.json`
- `evaluation/outputs/judge_results.json`
- `evaluation/outputs/bertscore_results.json`
- `evaluation/reports/summary_report.md`
