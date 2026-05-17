# PaperWeave Evaluation Report

Questions evaluated: 10
Pipeline runs: 30

## Leaderboard

| rank | pipeline | hackathon_weighted_score | avg_token_reduction_pct_vs_llm_only | avg_total_latency_ms | avg_bertscore_raw_f1 | avg_bertscore_rescaled_f1 | avg_judge_correctness_pct | judge_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | llm-only | 63.275 | 0.000 | 5850.007 | 0.507 | -0.016 | 50.000 | 0.000 |
| 2 | graphrag | 58.851 | 52.574 | 56954.970 | 0.586 | 0.146 | 72.000 | 0.800 |
| 3 | basic-rag | 54.533 | -379.083 | 6910.437 | 0.518 | 0.006 | 72.000 | 0.800 |

## Hackathon Criteria

- Token Reduction: 30% of weighted score, based on average total-token reduction relative to LLM-only.
- Answer Accuracy: 30%, using the strongest available signal among BERTScore and judge score.
- Performance / Latency: 20%, based on relative total latency.
- Engineering & Storytelling: 20%, based on retrieval hit rate, citation correctness, duplicate control, and fabricated citation avoidance.

## Pipeline Summary

| pipeline | count | failures | avg_total_tokens | p50_total_latency_ms | p95_total_latency_ms |
| --- | --- | --- | --- | --- | --- |
| basic-rag | 10 | 0 | 1159.300 | 5093.438 | 15670.037 |
| graphrag | 10 | 0 | 113.200 | 51410.761 | 89077.690 |
| llm-only | 10 | 0 | 257.000 | 4940.851 | 11164.869 |

## Hosted API Cost Estimates

Estimated using current hosted Qwen 2.5 7B pricing of `$0.040 / 1M input tokens` and `$0.100 / 1M output tokens`.

Source pricing reference:

- ComputePrices Qwen 2.5 7B: https://computeprices.com/models/qwen-2-5-7b

Approximate per-query generation cost:

| pipeline | avg_prompt_tokens | avg_output_tokens | estimated_cost_per_query_usd |
| --- | --- | --- | --- |
| llm-only | 40.8 | 216.2 | $0.000023 |
| basic-rag | 941.8 | 217.5 | $0.000059 |
| graphrag | 11.3 | 101.9 | $0.000011 |

Derived headline:

- GraphRAG token reduction vs Basic RAG: `90.24%`
- GraphRAG estimated API cost reduction vs Basic RAG: `82.09%`

Notes:

- These are hosted-API-equivalent estimates for submission reporting, not actual billed local-Ollama costs.
- Embedding costs are excluded because this benchmark run used local Ollama embeddings and the submission metric is dominated by answer-generation token spend.

## Bonus Checks

### BERTScore Setup

- Backend: `evaluate`
- Model: `microsoft/deberta-xlarge-mnli`
- Rescaled with baseline: `True`

### BERTScore

- basic-rag: bonus pass = False
- graphrag: bonus pass = False
- llm-only: bonus pass = False
### Judge Setup

- Metric: pass rate plus average correctness percentage

### Judge

- basic-rag: bonus pass = False, pass rate = 80.0%, avg correctness = 72.0%
- graphrag: bonus pass = False, pass rate = 80.0%, avg correctness = 72.0%
- llm-only: bonus pass = False, pass rate = 0.0%, avg correctness = 50.0%

## Outputs

- Raw benchmark records: `evaluation/outputs/benchmark_results.json`
- BERTScore results: `evaluation/outputs/bertscore_results.json`
- Judge results: `evaluation/outputs/judge_results.json`
- Leaderboard CSV: `evaluation/outputs/leaderboard.csv`
- Visualizations: `evaluation/reports/*.png`

## Known Limitations

- GraphRAG metrics depend on the running TigerGraph service exposing source snippets.
- Token counts are estimated with the local tokenizer when upstream providers do not return usage.
- Heuristic hallucination and retrieval metrics are useful diagnostics, not replacements for expert review.
