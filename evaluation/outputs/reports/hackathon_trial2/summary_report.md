# PaperWeave Evaluation Report

Questions evaluated: 5
Pipeline runs: 15

## Leaderboard

| rank | pipeline | hackathon_weighted_score | avg_token_reduction_pct_vs_llm_only | avg_total_latency_ms | avg_bertscore_raw_f1 | avg_bertscore_rescaled_f1 | avg_judge_correctness_pct | judge_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | graphrag | 72.686 | 87.328 | 67.817 | 0.394 | -0.250 | 40.000 | 0.000 |
| 2 | llm-only | 44.939 | 0.000 | 4157.771 | 0.499 | -0.032 | 32.000 | 0.000 |
| 3 | basic-rag | 31.520 | -363.451 | 4518.416 | 0.460 | -0.113 | 56.000 | 0.400 |

## Hackathon Criteria

- Token Reduction: 30% of weighted score, based on average total-token reduction relative to LLM-only.
- Answer Accuracy: 30%, using the strongest available signal among BERTScore and judge score.
- Performance / Latency: 20%, based on relative total latency.
- Engineering & Storytelling: 20%, based on retrieval hit rate, citation correctness, duplicate control, and fabricated citation avoidance.

## Pipeline Summary

| pipeline | count | failures | avg_total_tokens | p50_total_latency_ms | p95_total_latency_ms |
| --- | --- | --- | --- | --- | --- |
| basic-rag | 5 | 0 | 1061.800 | 4515.549 | 5403.127 |
| graphrag | 5 | 0 | 28.800 | 66.904 | 73.082 |
| llm-only | 5 | 0 | 235.000 | 4328.653 | 5015.618 |

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

- basic-rag: bonus pass = False, pass rate = 40.0%, avg correctness = 56.0%
- graphrag: bonus pass = False, pass rate = 0.0%, avg correctness = 40.0%
- llm-only: bonus pass = False, pass rate = 0.0%, avg correctness = 32.0%

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
