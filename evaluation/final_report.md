# PaperWeave Evaluation Report

Questions evaluated: 60
Pipeline runs: 180

## Leaderboard

| rank | pipeline | hackathon_weighted_score | avg_token_reduction_pct_vs_llm_only | avg_total_latency_ms | avg_bertscore_raw_f1 | avg_bertscore_rescaled_f1 | avg_judge_correctness_pct | judge_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | llm-only | 64.332 | 0.000 | 5345.732 | 0.448 | -0.139 | 33.333 | 0.000 |
| 2 | graphrag | 53.402 | -1.245 | 48205.078 | 0.513 | -0.004 | 41.333 | 0.067 |
| 3 | basic-rag | 44.445 | -1492.255 | 3701.211 | 0.466 | -0.101 | 40.333 | 0.033 |

## Hackathon Criteria

- Token Reduction: 30% of weighted score, based on average total-token reduction relative to LLM-only.
- Answer Accuracy: 30%, using the strongest available signal among BERTScore and judge score.
- Performance / Latency: 20%, based on relative total latency.
- Engineering & Storytelling: 20%, based on retrieval hit rate, citation correctness, duplicate control, and fabricated citation avoidance.

## Pipeline Summary

| pipeline | count | failures | avg_total_tokens | p50_total_latency_ms | p95_total_latency_ms |
| --- | --- | --- | --- | --- | --- |
| basic-rag | 60 | 0 | 1136.083 | 3593.043 | 5224.084 |
| graphrag | 60 | 0 | 72.217 | 51210.047 | 59291.907 |
| llm-only | 60 | 0 | 71.400 | 4843.983 | 7196.331 |

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

- basic-rag: bonus pass = False, pass rate = 3.3%, avg correctness = 40.3%
- graphrag: bonus pass = False, pass rate = 6.7%, avg correctness = 41.3%
- llm-only: bonus pass = False, pass rate = 0.0%, avg correctness = 33.3%

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
