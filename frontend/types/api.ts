export type TokenUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
};

export type TimingBreakdown = {
  retrieval_ms: number;
  generation_ms: number;
  evaluation_ms: number;
  total_ms: number;
};

export type SourceRecord = {
  id: string;
  title?: string | null;
  snippet: string;
  score?: number | null;
  metadata: Record<string, unknown>;
};

export type RetrievalInfo = {
  mode: string;
  top_k?: number | null;
  num_hops?: number | null;
  chunk_strategy?: string | null;
  graph_name?: string | null;
  raw: Record<string, unknown>;
};

export type EvaluationResult = {
  bertscore_f1?: number | null;
  judge_pass?: boolean | null;
  judge_reasoning?: string | null;
};

export type LivePipelineMetrics = {
  total_latency_ms: number;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  evaluation_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  token_reduction_pct_vs_llm_only: number;
  bertscore_raw_f1?: number | null;
  bertscore_rescaled_f1?: number | null;
  judge_score?: number | null;
  judge_correctness_pct?: number | null;
  judge_pass?: boolean | null;
  retrieval_quality: number;
  citation_correctness: number;
};

export type LiveJudgeResult = {
  score?: number | null;
  passed?: boolean | null;
  reasoning?: string | null;
  hallucination_level?: number | null;
  factual_correctness?: number | null;
  grounding?: number | null;
  completeness?: number | null;
  scientific_accuracy?: number | null;
};

export type HallucinationResult = {
  fabricated_citation_count: number;
  fabricated_citation_rate: number;
  answer_context_mismatch: number;
  unsupported_claim_estimate: number;
  has_fabricated_citations: boolean;
  high_answer_context_mismatch: boolean;
  high_unsupported_claim_risk: boolean;
};

export type RetrievalQualityResult = {
  retrieval_hit: boolean;
  retrieved_chunk_count: number;
  source_overlap: number;
  citation_correctness: number;
  context_relevance: number;
  useful_chunk_ratio: number;
  duplicate_chunk_ratio: number;
};

export type LivePipelineResult = {
  answer: string;
  tokens: TokenUsage;
  latency: number;
  estimated_cost: number;
  sources: SourceRecord[];
  retrieval_info: RetrievalInfo;
  timing_breakdown: TimingBreakdown;
  metrics: LivePipelineMetrics;
  judge: LiveJudgeResult;
  hallucination: HallucinationResult;
  retrieval: RetrievalQualityResult;
  evaluation_reference?: string | null;
  raw: Record<string, unknown>;
};

export type AskResponse = {
  pipeline: string;
  answer: string;
  tokens: TokenUsage;
  latency: number;
  estimated_cost: number;
  sources: SourceRecord[];
  retrieval_info: RetrievalInfo;
  timing_breakdown: TimingBreakdown;
  evaluation?: EvaluationResult | null;
};

export type AskAllResponse = {
  question: string;
  pipelines: Record<string, LivePipelineResult>;
  leaderboard: Array<Record<string, number | string | boolean | null>>;
  global_metrics: Record<string, number | string | boolean | null>;
  llm_only?: AskResponse;
  basic_rag?: AskResponse;
  graphrag?: AskResponse;
  errors?: Record<string, string>;
};
