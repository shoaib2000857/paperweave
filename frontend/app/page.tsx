"use client";

import { useState } from "react";

import { AppNav } from "@/components/app-nav";
import { ComparisonChart } from "@/components/comparison-chart";
import { GraphPreview } from "@/components/graph-preview";
import { QueryForm } from "@/components/query-form";
import { ResultCard } from "@/components/result-card";
import { SourceList } from "@/components/source-list";
import { askAll } from "@/lib/api";
import { AskAllResponse } from "@/types/api";

type Phase = "idle" | "running" | "evaluating" | "done";

const PHASE_LABEL: Record<Phase, string> = {
  idle: "",
  running: "Running pipelines: LLM-only · Basic RAG · TigerGraph GraphRAG",
  evaluating: "Computing metrics: BERTScore · Judge LLM · Hallucination · Retrieval quality",
  done: "",
};

export default function HomePage() {
  const [result, setResult] = useState<AskAllResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");

  async function handleSubmit(question: string, referenceAnswer?: string) {
    setError(null);
    setResult(null);
    setPhase("running");

    const evalTimer = setTimeout(() => setPhase("evaluating"), 4000);

    try {
      const response = await askAll(question, referenceAnswer);
      clearTimeout(evalTimer);
      setResult(response);
      setPhase("done");
    } catch (err) {
      clearTimeout(evalTimer);
      setError(err instanceof Error ? err.message : "Unknown error");
      setPhase("idle");
    }
  }

  const isLoading = phase === "running" || phase === "evaluating";

  const chartData = result
    ? [
        {
          name: "LLM-only",
          tokens: result.pipelines["llm-only"]?.metrics.total_tokens ?? 0,
          latency: result.pipelines["llm-only"]?.metrics.total_latency_ms ?? 0,
          bertscore: result.pipelines["llm-only"]?.metrics.bertscore_rescaled_f1 ?? 0,
          judge: result.pipelines["llm-only"]?.metrics.judge_score ?? 0,
          hallucination: result.pipelines["llm-only"]?.hallucination.fabricated_citation_rate ?? 0,
        },
        {
          name: "Basic RAG",
          tokens: result.pipelines["basic-rag"]?.metrics.total_tokens ?? 0,
          latency: result.pipelines["basic-rag"]?.metrics.total_latency_ms ?? 0,
          bertscore: result.pipelines["basic-rag"]?.metrics.bertscore_rescaled_f1 ?? 0,
          judge: result.pipelines["basic-rag"]?.metrics.judge_score ?? 0,
          hallucination: result.pipelines["basic-rag"]?.hallucination.fabricated_citation_rate ?? 0,
        },
        {
          name: "GraphRAG",
          tokens: result.pipelines["graphrag"]?.metrics.total_tokens ?? 0,
          latency: result.pipelines["graphrag"]?.metrics.total_latency_ms ?? 0,
          bertscore: result.pipelines["graphrag"]?.metrics.bertscore_rescaled_f1 ?? 0,
          judge: result.pipelines["graphrag"]?.metrics.judge_score ?? 0,
          hallucination: result.pipelines["graphrag"]?.hallucination.fabricated_citation_rate ?? 0,
        },
      ]
    : [];

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-8 px-4 py-8 md:px-6 lg:px-8">
      <header className="rounded-[2.5rem] border border-white/70 bg-white/75 p-8 shadow-panel backdrop-blur">
        <div className="mb-6">
          <AppNav active="home" />
        </div>
        <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3 inline-flex rounded-full bg-ember px-4 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-paper">
              PaperWeave
            </div>
            <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">
              Graph-grounded benchmarking for scientific paper reasoning
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-ink/75 md:text-base">
              Compare LLM-only, Basic RAG, and TigerGraph GraphRAG on the same question. Evaluation metrics compute live on each generated answer.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <Stat label="Pipelines" value="3" />
            <Stat label="Domain" value="NLP / LLM papers" />
            <Stat label="Core metric" value="Token reduction" />
            <Stat label="Backend" value="TigerGraph GraphRAG" />
          </div>
        </div>
      </header>

      <QueryForm onSubmit={handleSubmit} />

      {isLoading ? (
        <div className="rounded-[1.5rem] border border-amber-200/60 bg-amber-50/80 px-6 py-4 shadow-panel backdrop-blur">
          <div className="flex items-center gap-3">
            <span className="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-ember" />
            <span className="text-sm font-medium text-amber-900/80">{PHASE_LABEL[phase]}</span>
          </div>
        </div>
      ) : null}

      {error ? <div className="rounded-[1.5rem] bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div> : null}

      <section className="grid gap-6 xl:grid-cols-3">
        <ResultCard title="LLM-only Baseline" result={result?.pipelines["llm-only"]} loading={isLoading} />
        <ResultCard title="Basic RAG" result={result?.pipelines["basic-rag"]} loading={isLoading} />
        <ResultCard title="TigerGraph GraphRAG" result={result?.pipelines["graphrag"]} loading={isLoading} />
      </section>

      {result?.leaderboard?.length ? (
        <section className="rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-panel backdrop-blur">
          <div className="mb-1 flex items-center gap-3">
            <h2 className="text-lg font-semibold">Live Leaderboard</h2>
            <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[0.6rem] font-semibold uppercase tracking-[0.2em] text-emerald-700">
              Live
            </span>
          </div>
          {result.global_metrics ? (
            <p className="mb-4 text-xs text-ink/45">
              Reference:{" "}
              {String(result.global_metrics.evaluation_reference_source ?? "cross_pipeline_consensus").replace(/_/g, " ")}
              {result.global_metrics.evaluation_runtime_ms
                ? ` · Eval runtime: ${Number(result.global_metrics.evaluation_runtime_ms).toFixed(0)} ms`
                : null}
              {result.global_metrics.best_pipeline ? ` · Best: ${String(result.global_metrics.best_pipeline)}` : null}
            </p>
          ) : null}
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-ink/10 text-left text-xs uppercase tracking-[0.18em] text-ink/50">
                  <th className="pb-2 pr-4">Rank</th>
                  <th className="pb-2 pr-4">Pipeline</th>
                  <th className="pb-2 pr-4">Weighted</th>
                  <th className="pb-2 pr-4">BERTScore</th>
                  <th className="pb-2 pr-4">Judge Pass</th>
                  <th className="pb-2 pr-4">Latency</th>
                  <th className="pb-2 pr-4">Token Δ</th>
                </tr>
              </thead>
              <tbody>
                {result.leaderboard.map((row, index) => (
                  <tr
                    key={`${row.pipeline}-${index}`}
                    className={`border-b border-ink/5 ${index === 0 ? "bg-amber-50/60" : ""}`}
                  >
                    <td className="py-2 pr-4 font-medium">#{String(row.rank ?? index + 1)}</td>
                    <td className="py-2 pr-4 font-medium">
                      {String(row.pipeline ?? "unknown")}
                      {index === 0 ? (
                        <span className="ml-2 rounded-full bg-amber-200/60 px-2 py-0.5 text-[0.6rem] text-amber-800">
                          Best
                        </span>
                      ) : null}
                    </td>
                    <td className="py-2 pr-4">{Number(row.hackathon_weighted_score ?? 0).toFixed(2)}</td>
                    <td className="py-2 pr-4">{Number(row.avg_bertscore_rescaled_f1 ?? 0).toFixed(3)}</td>
                    <td className="py-2 pr-4">{(Number(row.judge_pass_rate ?? 0) * 100).toFixed(1)}%</td>
                    <td className="py-2 pr-4">{Number(row.avg_total_latency_ms ?? 0).toFixed(0)} ms</td>
                    <td className="py-2 pr-4">{Number(row.avg_token_reduction_pct_vs_llm_only ?? 0).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="grid gap-6 lg:grid-cols-[1.2fr,0.8fr]">
        <ComparisonChart data={chartData} />
        <GraphPreview sources={result?.pipelines["graphrag"]?.sources ?? result?.graphrag?.sources ?? []} />
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <SourceList title="Basic RAG Evidence" sources={result?.pipelines["basic-rag"]?.sources ?? []} />
        <SourceList title="GraphRAG Evidence" sources={result?.pipelines["graphrag"]?.sources ?? []} />
      </section>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1.25rem] bg-paper/90 p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-ink/45">{label}</div>
      <div className="mt-1 text-base font-semibold">{value}</div>
    </div>
  );
}
