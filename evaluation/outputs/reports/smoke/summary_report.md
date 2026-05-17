# PaperWeave Evaluation Report

Questions evaluated: 1
Pipeline runs: 3

## Leaderboard

| rank | pipeline | hackathon_weighted_score | avg_token_reduction_pct_vs_llm_only | avg_total_latency_ms | avg_bertscore_rescaled_f1 | judge_pass_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | basic-rag | 64.783 | 100.000 | 1.258 | 0.000 | 0.000 |
| 2 | llm-only | 45.000 | 0.000 | 115.818 | 0.000 | 0.000 |
| 3 | graphrag | 32.941 | -1800.000 | 11.926 | 0.000 | 0.000 |

## Hackathon Criteria

- Token Reduction: 30% of weighted score, based on average total-token reduction relative to LLM-only.
- Answer Accuracy: 30%, using the strongest available signal among BERTScore and judge score.
- Performance / Latency: 20%, based on relative total latency.
- Engineering & Storytelling: 20%, based on retrieval hit rate, citation correctness, duplicate control, and fabricated citation avoidance.

## Pipeline Summary

| pipeline | count | failures | avg_total_tokens | p50_total_latency_ms | p95_total_latency_ms |
| --- | --- | --- | --- | --- | --- |
| basic-rag | 1 | 0 | 0.000 | 1.258 | 1.258 |
| graphrag | 1 | 0 | 19.000 | 11.926 | 11.926 |
| llm-only | 1 | 1 | 0.000 | 115.818 | 115.818 |

## Bonus Checks


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
