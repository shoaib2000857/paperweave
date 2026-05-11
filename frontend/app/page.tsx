"use client";

import { useState } from "react";

import { ComparisonChart } from "@/components/comparison-chart";
import { GraphPreview } from "@/components/graph-preview";
import { QueryForm } from "@/components/query-form";
import { ResultCard } from "@/components/result-card";
import { SourceList } from "@/components/source-list";
import { askAll } from "@/lib/api";
import { AskAllResponse } from "@/types/api";

export default function HomePage() {
  const [result, setResult] = useState<AskAllResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(question: string) {
    setError(null);
    try {
      const response = await askAll(question);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  }

  const chartData = result
    ? [
        { name: "LLM-only", tokens: result.llm_only.tokens.total_tokens, latency: result.llm_only.latency },
        { name: "Basic RAG", tokens: result.basic_rag.tokens.total_tokens, latency: result.basic_rag.latency },
        { name: "GraphRAG", tokens: result.graphrag.tokens.total_tokens, latency: result.graphrag.latency },
      ]
    : [];

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-8 px-4 py-8 md:px-6 lg:px-8">
      <header className="rounded-[2.5rem] border border-white/70 bg-white/75 p-8 shadow-panel backdrop-blur">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3 inline-flex rounded-full bg-ember px-4 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-paper">
              PaperWeave
            </div>
            <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">
              Graph-grounded benchmarking for scientific paper reasoning
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-ink/75 md:text-base">
              Compare `LLM-only`, `Basic RAG`, and `TigerGraph GraphRAG` on the same question, with direct visibility into tokens, latency, cost, and retrieval evidence.
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
      {error ? <div className="rounded-[1.5rem] bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div> : null}

      <section className="grid gap-6 xl:grid-cols-3">
        <ResultCard title="LLM-only Baseline" result={result?.llm_only} />
        <ResultCard title="Basic RAG" result={result?.basic_rag} />
        <ResultCard title="TigerGraph GraphRAG" result={result?.graphrag} />
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.2fr,0.8fr]">
        <ComparisonChart data={chartData} />
        <GraphPreview sources={result?.graphrag.sources ?? []} />
      </section>

      <SourceList sources={result?.graphrag.sources ?? []} />
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
