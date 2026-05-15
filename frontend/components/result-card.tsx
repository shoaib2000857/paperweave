import { LivePipelineResult } from "@/types/api";

type Props = {
  title: string;
  result?: LivePipelineResult;
  loading?: boolean;
};

export function ResultCard({ title, result, loading = false }: Props) {
  const statusLabel = loading ? "computing" : result ? "live" : "idle";
  const bertscoreLabel = result?.evaluation_reference === "user_reference_answer" ? "BERTScore" : "BERTSim";
  const statusClass = loading
    ? "bg-amber-200 text-amber-900"
    : result
      ? "bg-emerald-100 text-emerald-800"
      : "bg-ink text-paper";

  return (
    <section className="rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-panel backdrop-blur">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">{title}</h2>
        <span className={`rounded-full px-3 py-1 text-xs uppercase tracking-[0.2em] ${statusClass}`}>
          {statusLabel}
        </span>
      </div>

      {loading ? (
        <div className="flex min-h-40 flex-col items-center justify-center gap-3">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-ember" />
          <span className="text-xs text-ink/40">Generating answer · computing evaluation metrics...</span>
        </div>
      ) : (
        <p className="min-h-40 text-sm leading-7 text-ink/85">
          {result?.answer ?? "Run a query to render this pipeline's answer, token usage, timing breakdown, and evidence."}
        </p>
      )}

      <div className="mt-6 grid grid-cols-2 gap-3 text-sm">
        <Metric label="Tokens" value={result ? String(result.metrics.total_tokens) : "-"} />
        <Metric label="Latency" value={result ? `${result.metrics.total_latency_ms.toFixed(0)} ms` : "-"} />
        <Metric label="Cost" value={result ? `$${result.estimated_cost.toFixed(5)}` : "-"} />
        <Metric label="Sources" value={result ? String(result.sources.length) : "-"} />
        <Metric
          label={bertscoreLabel}
          value={result ? (result.metrics.bertscore_rescaled_f1 ?? 0).toFixed(3) : "-"}
        />
        <Metric
          label="Judge"
          value={
            result
              ? `${(result.metrics.judge_correctness_pct ?? (result.metrics.judge_score ?? 0) * 20).toFixed(0)}% (${result.metrics.judge_pass ? "pass" : "fail"})`
              : "-"
          }
        />
        <Metric
          label="Hallucination"
          value={result ? `${(result.hallucination.fabricated_citation_rate * 100).toFixed(1)}% fabricated` : "-"}
        />
        <Metric label="Retrieval quality" value={result ? result.retrieval.context_relevance.toFixed(3) : "-"} />
      </div>

      {result?.judge.reasoning ? (
        <details className="mt-4">
          <summary className="cursor-pointer text-xs text-ink/40 hover:text-ink/60">Judge reasoning</summary>
          <p className="mt-2 rounded-[1rem] bg-paper/80 px-4 py-3 text-xs leading-6 text-ink/65">{result.judge.reasoning}</p>
        </details>
      ) : null}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1.25rem] bg-paper/90 p-3">
      <div className="text-xs uppercase tracking-[0.18em] text-ink/45">{label}</div>
      <div className="mt-1 text-base font-semibold">{value}</div>
    </div>
  );
}
