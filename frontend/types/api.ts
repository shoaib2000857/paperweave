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
  llm_only: AskResponse;
  basic_rag: AskResponse;
  graphrag: AskResponse;
};
