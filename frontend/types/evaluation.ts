import { SourceRecord } from "@/types/api";

export type PipelineName = "llm-only" | "basic-rag" | "graphrag";

export type BenchmarkRecord = {
  question_id: string;
  question: string;
  ground_truth: string;
  category: string;
  difficulty?: string;
  expected_sources?: string[];
  pipeline: PipelineName;
  answer: string;
  retrieved_context?: string;
  sources: SourceRecord[];
  source_titles: string[];
  prompt_tokens: number;
  retrieved_context_tokens: number;
  output_tokens: number;
  total_tokens: number;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  evaluation_latency_ms: number;
  total_latency_ms: number;
  retrieval_mode: string;
  retrieved_chunk_count: number;
  duplicate_chunk_ratio: number;
  raw: Record<string, unknown>;
  error: string | null;
  token_reduction_pct_vs_llm_only: number;
  answer_token_efficiency: number;
  retrieval_compression_efficiency: number;
  prompt_context_share: number;
  source_overlap: number;
  retrieval_hit: boolean;
  citation_correctness: number;
  context_relevance: number;
  useful_chunk_ratio: number;
  fabricated_citation_count: number;
  fabricated_citation_rate: number;
  answer_context_mismatch: number;
  unsupported_claim_estimate: number;
  bertscore_raw_f1?: number;
  bertscore_rescaled_f1?: number;
  judge_score?: number;
  judge_correctness_pct?: number;
  judge_pass?: boolean;
};

export type PipelineSummary = {
  pipeline: string;
  count?: number;
  failures?: number;
  avg_total_tokens?: number;
  avg_token_reduction_pct_vs_llm_only?: number;
  avg_total_latency_ms?: number;
  p50_total_latency_ms?: number;
  p95_total_latency_ms?: number;
  avg_bertscore_raw_f1?: number;
  avg_bertscore_rescaled_f1?: number;
  judge_pass_rate?: number;
  avg_judge_score?: number;
  avg_judge_correctness_pct?: number;
  judge_hallucination_rate?: number;
  avg_fabricated_citation_rate?: number;
  avg_answer_context_mismatch?: number;
  avg_duplicate_chunk_ratio?: number;
  avg_source_overlap?: number;
  avg_citation_correctness?: number;
  avg_context_relevance?: number;
  avg_useful_chunk_ratio?: number;
  avg_unsupported_claim_estimate?: number;
  hackathon_token_reduction_score?: number;
  hackathon_answer_accuracy_score?: number;
  hackathon_performance_latency_score?: number;
  hackathon_engineering_storytelling_score?: number;
  hackathon_weighted_score?: number;
  [key: string]: number | string | undefined;
};

export type EvaluationLeaderboardRow = PipelineSummary & {
  rank: number;
};

export type EvaluationArtifact = {
  available: boolean;
  path: string;
  data: Record<string, unknown> | null;
};

export type EvaluationReportChart = {
  name: string;
  path: string;
  url: string;
};

export type EvaluationReport = {
  available: boolean;
  path: string;
  markdown: string | null;
  charts: EvaluationReportChart[];
};

export type EvaluationBenchmark = {
  dataset: string;
  question_count: number;
  pipelines: PipelineName[];
  top_k: number | null;
  records: BenchmarkRecord[];
  summary: Record<string, PipelineSummary>;
};

export type EvaluationOfflineSnapshot = {
  benchmark: EvaluationBenchmark;
  leaderboard: EvaluationLeaderboardRow[];
};

export type EvaluationDashboardResponse = {
  benchmark: EvaluationBenchmark;
  leaderboard: EvaluationLeaderboardRow[];
  bertscore: EvaluationArtifact;
  judge: EvaluationArtifact;
  report: EvaluationReport;
  live?: Record<string, unknown> | null;
  offline?: EvaluationOfflineSnapshot;
};
