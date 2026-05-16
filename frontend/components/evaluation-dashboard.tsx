"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { AppNav } from "@/components/app-nav";
import { apiUrl, fetchEvaluationDashboard } from "@/lib/evaluation";
import {
  BenchmarkRecord,
  EvaluationArtifact,
  EvaluationBenchmark,
  EvaluationDashboardResponse,
  EvaluationLeaderboardRow,
  EvaluationReportChart,
  PipelineSummary,
} from "@/types/evaluation";

type SortKey = "weighted" | "latency" | "tokenReduction" | "bertscore" | "judgeCorrectness";
type DataSource = "active" | "offline";

const PIPELINE_ORDER = ["llm-only", "basic-rag", "graphrag"] as const;

export function EvaluationDashboard() {
  const [payload, setPayload] = useState<EvaluationDashboardResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>("weighted");
  const [dataSource, setDataSource] = useState<DataSource>("active");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const response = await fetchEvaluationDashboard();
        if (!active) {
          return;
        }
        setPayload(response);
        setError(null);
      } catch (fetchError) {
        if (!active) {
          return;
        }
        setError(fetchError instanceof Error ? fetchError.message : "Failed to load evaluation dashboard");
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    load();
    const interval = setInterval(load, 12000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const hasLive = Boolean(payload?.live && payload?.leaderboard?.length);
  const selectedDataSource: DataSource = dataSource === "offline" || !hasLive ? "offline" : "active";
  const activeBenchmark = selectedDataSource === "active" ? payload?.benchmark : payload?.offline?.benchmark;
  const activeLeaderboard = selectedDataSource === "active" ? payload?.leaderboard ?? [] : payload?.offline?.leaderboard ?? [];
  const summaries = useMemo(
    () => Object.entries(activeBenchmark?.summary ?? {}).map(([pipeline, summary]) => ({ ...summary, pipeline })),
    [activeBenchmark],
  );

  const sortedLeaderboard = useMemo(() => {
    const rows = [...activeLeaderboard];
    rows.sort((left, right) => leaderboardSortValue(right, sortKey) - leaderboardSortValue(left, sortKey));
    return rows;
  }, [activeLeaderboard, sortKey]);

  const questionGroups = useMemo(() => groupQuestions(activeBenchmark?.records ?? []), [activeBenchmark]);
  const metricCards = useMemo(
    () => buildMetricCards(sortedLeaderboard, summaries, activeBenchmark, selectedDataSource),
    [sortedLeaderboard, summaries, activeBenchmark, selectedDataSource],
  );
  const chartRows = useMemo(() => buildChartRows(summaries), [summaries]);
  const reportPreview = payload?.report.markdown ? payload.report.markdown.split("\n").slice(0, 18).join("\n") : null;
  const artifactCards = buildArtifactCards(payload?.bertscore, payload?.judge, payload?.report);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_top_left,rgba(244,169,92,0.16),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(36,97,92,0.18),transparent_32%),linear-gradient(180deg,rgba(8,15,25,0.96),rgba(2,6,23,1))]" />
      <div className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-8 px-4 py-6 md:px-6 lg:px-8">
        <header className="rounded-[2.25rem] border border-white/10 bg-white/5 p-6 shadow-[0_30px_100px_rgba(0,0,0,0.28)] backdrop-blur-xl md:p-8">
          <div className="flex flex-col gap-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="max-w-3xl space-y-4">
                <AppNav active="evaluation" />
                <div className="flex flex-wrap items-center gap-3">
                  <div className="inline-flex rounded-full border border-emerald-400/20 bg-emerald-400/10 px-4 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.32em] text-emerald-200">
                    Accuracy evaluation
                  </div>
                  {hasLive ? (
                    <div className="inline-flex items-center gap-2 rounded-full border border-sky-400/25 bg-sky-400/10 px-4 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.28em] text-sky-200">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" />
                      Live + offline
                    </div>
                  ) : (
                    <div className="inline-flex rounded-full border border-slate-500/30 bg-slate-700/40 px-4 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.28em] text-slate-400">
                      Offline benchmark only
                    </div>
                  )}
                </div>
                <div className="space-y-3">
                  <h1 className="text-4xl font-semibold tracking-tight text-white md:text-5xl">
                    Benchmark answers, not just tokens
                  </h1>
                  <p className="max-w-2xl text-sm leading-7 text-slate-300 md:text-base">
                    This page shows the hackathon evaluation outputs for LLM-only, Basic RAG, and TigerGraph GraphRAG using both judge-based grading and BERTScore.
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <Stat label="Dataset" value={String(activeBenchmark?.dataset ?? "n/a")} />
                <Stat label="Questions" value={String(activeBenchmark?.question_count ?? 0)} />
                <Stat label="Source" value={selectedDataSource === "active" ? "Active view" : "Offline view"} />
                <Stat label="Charts" value={String(payload?.report.charts.length ?? 0)} />
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              {metricCards.map((card) => (
                <MetricCard key={card.label} {...card} />
              ))}
            </div>
          </div>
        </header>

        {loading ? <LoadingState /> : null}
        {error ? <ErrorState message={error} /> : null}

        {!loading && !error && payload ? (
          <>
            <section className="grid gap-6 xl:grid-cols-[1.15fr,0.85fr]">
              <SectionCard title="Evaluation snapshot" description="Choose between the latest live query evaluation and the saved offline benchmark outputs.">
                <div className="mb-5 flex flex-wrap items-center gap-3">
                  <ToggleButton
                    active={selectedDataSource === "active"}
                    disabled={!hasLive}
                    onClick={() => setDataSource("active")}
                    label="Active benchmark view"
                  />
                  <ToggleButton active={selectedDataSource === "offline"} onClick={() => setDataSource("offline")} label="Offline benchmark view" />
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <SnapshotCard
                    label="Dataset path"
                    value={String(activeBenchmark?.dataset ?? "unavailable")}
                    detail={`${activeBenchmark?.question_count ?? 0} questions · ${(activeBenchmark?.pipelines ?? []).join(" · ") || "No pipelines"}`}
                  />
                  <SnapshotCard
                    label="Top K"
                    value={activeBenchmark?.top_k == null ? "default" : String(activeBenchmark.top_k)}
                    detail={selectedDataSource === "active" ? "Live query replay bundle" : "Saved benchmark artifact"}
                  />
                </div>
              </SectionCard>

              <SectionCard title="Artifacts" description="Direct visibility into the benchmark result files and generated report assets.">
                <div className="grid gap-4 md:grid-cols-3">
                  {artifactCards.map((artifact) => (
                    <ArtifactCard key={artifact.label} {...artifact} />
                  ))}
                </div>
              </SectionCard>
            </section>

            <section className="grid gap-6">
              <SectionCard title="Leaderboard" description="Weighted ranking plus the raw accuracy signals the hackathon expects.">
                <div className="mb-4 flex flex-wrap items-center gap-3">
                  <label className="text-xs uppercase tracking-[0.22em] text-slate-400">Sort by</label>
                  <select
                    value={sortKey}
                    onChange={(event) => setSortKey(event.target.value as SortKey)}
                    className="rounded-full border border-white/10 bg-slate-900 px-4 py-2 text-sm text-slate-100 outline-none transition focus:border-amber-300/60"
                  >
                    <option value="weighted">Weighted score</option>
                    <option value="latency">Fastest latency</option>
                    <option value="tokenReduction">Token reduction</option>
                    <option value="bertscore">BERTScore rescaled F1</option>
                    <option value="judgeCorrectness">Judge correctness %</option>
                  </select>
                </div>
                <LeaderboardTable rows={sortedLeaderboard} highlightedPipeline={sortedLeaderboard[0]?.pipeline ?? ""} />
              </SectionCard>
            </section>

            <section className="grid gap-6 lg:grid-cols-2">
              <SectionCard title="Pipeline signals" description="Quick visual comparison of token, latency, and accuracy behavior.">
                <div className="space-y-6">
                  <MetricChart title="Average total tokens" data={chartRows} dataKey="avg_total_tokens" color="#f59e0b" />
                  <MetricChart title="Average latency" data={chartRows} dataKey="avg_total_latency_ms" color="#14b8a6" />
                  <MetricChart title="Rescaled BERTScore F1" data={chartRows} dataKey="avg_bertscore_rescaled_f1" color="#60a5fa" />
                  <MetricChart title="Judge correctness %" data={chartRows} dataKey="avg_judge_correctness_pct" color="#f97316" />
                </div>
              </SectionCard>

              <SectionCard title="Report preview" description="Front-end preview of the generated markdown report and exported chart files.">
                {payload.report.charts.length ? (
                  <div className="mb-5 grid gap-3 sm:grid-cols-2">
                    {payload.report.charts.map((chart) => (
                      <ChartCard key={chart.path} chart={chart} />
                    ))}
                  </div>
                ) : (
                  <div className="mb-5 rounded-[1.25rem] border border-white/10 bg-slate-950/60 p-4 text-sm text-slate-400">
                    No report charts generated yet. Run `python scripts/run_benchmark.py --judge`.
                  </div>
                )}
                <pre className="overflow-x-auto rounded-[1.25rem] border border-white/10 bg-slate-950/70 p-4 text-xs leading-6 text-slate-300">
                  {reportPreview ?? "No markdown report available yet."}
                </pre>
              </SectionCard>
            </section>

            <section className="grid gap-6">
              <SectionCard title="Question-by-question comparison" description="Inspect each answer, its retrieval evidence, and the evaluation warnings side by side.">
                <div className="space-y-6">
                  {questionGroups.length ? (
                    questionGroups.map((group) => <QuestionComparisonCard key={group.question_id} questionGroup={group} />)
                  ) : (
                    <div className="rounded-[1.5rem] border border-white/10 bg-slate-950/60 p-5 text-sm text-slate-400">
                      No benchmark records available yet. Run the benchmark script first.
                    </div>
                  )}
                </div>
              </SectionCard>
            </section>
          </>
        ) : null}
      </div>
    </main>
  );
}

function buildMetricCards(
  rows: EvaluationLeaderboardRow[],
  summaries: Array<PipelineSummary & { pipeline: string }>,
  benchmark: EvaluationBenchmark | undefined,
  dataSource: DataSource,
) {
  const best = rows[0];
  const avgTokenReduction = average(summaries.map((summary) => summary.avg_token_reduction_pct_vs_llm_only ?? 0));
  const avgJudgePassRate = average(summaries.map((summary) => (summary.judge_pass_rate ?? 0) * 100));
  const avgBertscore = average(summaries.map((summary) => summary.avg_bertscore_rescaled_f1 ?? 0));

  return [
    { label: "Leading pipeline", value: best?.pipeline ?? "n/a", detail: best ? `${formatNumber(best.hackathon_weighted_score)} weighted` : "No data yet" },
    { label: "Judge pass rate", value: `${avgJudgePassRate.toFixed(1)}%`, detail: "Average across pipelines" },
    { label: "Rescaled BERTScore", value: avgBertscore.toFixed(3), detail: "Semantic alignment to ground truth" },
    { label: "Avg token reduction", value: `${avgTokenReduction.toFixed(1)}%`, detail: "Relative to LLM-only" },
    {
      label: "Questions loaded",
      value: String(benchmark?.question_count ?? 0),
      detail: dataSource === "active" ? "Current live benchmark bundle" : "Saved offline evaluation file",
    },
  ];
}

function buildChartRows(summaries: Array<PipelineSummary & { pipeline: string }>) {
  return summaries
    .slice()
    .sort(
      (left, right) =>
        PIPELINE_ORDER.indexOf(left.pipeline as typeof PIPELINE_ORDER[number]) -
        PIPELINE_ORDER.indexOf(right.pipeline as typeof PIPELINE_ORDER[number]),
    )
    .map((summary) => ({
      name: summary.pipeline,
      avg_total_tokens: summary.avg_total_tokens ?? 0,
      avg_total_latency_ms: summary.avg_total_latency_ms ?? 0,
      avg_bertscore_rescaled_f1: summary.avg_bertscore_rescaled_f1 ?? 0,
      avg_judge_correctness_pct: summary.avg_judge_correctness_pct ?? ((summary.avg_judge_score ?? 0) / 5) * 100,
    }));
}

function groupQuestions(records: BenchmarkRecord[]) {
  const buckets = new Map<string, BenchmarkRecord[]>();
  for (const record of records) {
    const current = buckets.get(record.question_id) ?? [];
    current.push(record);
    buckets.set(record.question_id, current);
  }
  return Array.from(buckets.entries()).map(([questionId, items]) => ({
    question_id: questionId,
    question: items[0]?.question ?? questionId,
    category: items[0]?.category ?? "",
    difficulty: items[0]?.difficulty ?? "",
    records: items.slice().sort((left, right) => PIPELINE_ORDER.indexOf(left.pipeline) - PIPELINE_ORDER.indexOf(right.pipeline)),
  }));
}

function buildArtifactCards(
  bertscore: EvaluationArtifact | undefined,
  judge: EvaluationArtifact | undefined,
  report: EvaluationDashboardResponse["report"] | undefined,
) {
  return [
    {
      label: "BERTScore artifact",
      available: Boolean(bertscore?.available),
      detail: bertscore?.available
        ? `${String((bertscore.data?.backend as string | undefined) ?? "unknown")} backend`
        : "Not generated yet",
      path: bertscore?.path ?? "",
    },
    {
      label: "Judge artifact",
      available: Boolean(judge?.available),
      detail: judge?.available ? "Pass rate + correctness outputs available" : "Not generated yet",
      path: judge?.path ?? "",
    },
    {
      label: "Report bundle",
      available: Boolean(report?.available),
      detail: `${report?.charts.length ?? 0} chart exports`,
      path: report?.path ?? "",
    },
  ];
}

function leaderboardSortValue(row: EvaluationLeaderboardRow, sortKey: SortKey): number {
  if (sortKey === "latency") {
    return -(row.avg_total_latency_ms ?? 0);
  }
  if (sortKey === "tokenReduction") {
    return row.avg_token_reduction_pct_vs_llm_only ?? 0;
  }
  if (sortKey === "bertscore") {
    return row.avg_bertscore_rescaled_f1 ?? 0;
  }
  if (sortKey === "judgeCorrectness") {
    return row.avg_judge_correctness_pct ?? ((row.avg_judge_score ?? 0) / 5) * 100;
  }
  return row.hackathon_weighted_score ?? 0;
}

function formatNumber(value?: number) {
  return (value ?? 0).toFixed(3);
}

function formatLatency(value?: number) {
  return `${(value ?? 0).toFixed(0)} ms`;
}

function formatPercent(value?: number) {
  return `${(value ?? 0).toFixed(1)}%`;
}

function formatTokens(value?: number) {
  return `${Math.round(value ?? 0)} tok`;
}

function average(values: number[]) {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function LoadingState() {
  return <div className="rounded-[1.5rem] border border-white/10 bg-slate-900/70 p-6 text-sm text-slate-300">Loading evaluation results...</div>;
}

function ErrorState({ message }: { message: string }) {
  return <div className="rounded-[1.5rem] border border-rose-400/20 bg-rose-400/10 p-6 text-sm text-rose-100">{message}</div>;
}

function SectionCard({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-[0_24px_80px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <div className="mb-5 space-y-2">
        <h2 className="text-xl font-semibold text-white">{title}</h2>
        <p className="max-w-3xl text-sm leading-7 text-slate-400">{description}</p>
      </div>
      {children}
    </section>
  );
}

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-[1.75rem] border border-white/10 bg-slate-900/75 p-5 shadow-[0_18px_50px_rgba(0,0,0,0.24)]">
      <div className="text-[0.65rem] uppercase tracking-[0.3em] text-slate-400">{label}</div>
      <div className="mt-3 text-2xl font-semibold text-white">{value}</div>
      <div className="mt-2 text-sm leading-6 text-slate-400">{detail}</div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1.4rem] border border-white/10 bg-slate-900/70 p-4">
      <div className="text-[0.65rem] uppercase tracking-[0.28em] text-slate-400">{label}</div>
      <div className="mt-2 line-clamp-2 text-base font-semibold text-white">{value}</div>
    </div>
  );
}

function ToggleButton({
  active,
  disabled,
  onClick,
  label,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`rounded-full border px-4 py-2 text-sm transition ${
        active ? "border-amber-300/70 bg-amber-300/15 text-amber-100" : "border-white/10 bg-slate-900 text-slate-300"
      } ${disabled ? "cursor-not-allowed opacity-40" : "hover:border-amber-300/40"}`}
    >
      {label}
    </button>
  );
}

function SnapshotCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-[1.4rem] border border-white/10 bg-slate-950/60 p-4">
      <div className="text-[0.65rem] uppercase tracking-[0.28em] text-slate-400">{label}</div>
      <div className="mt-2 break-all text-sm font-semibold text-white">{value}</div>
      <div className="mt-2 text-sm leading-6 text-slate-400">{detail}</div>
    </div>
  );
}

function ArtifactCard({ label, available, detail, path }: { label: string; available: boolean; detail: string; path: string }) {
  return (
    <div className="rounded-[1.4rem] border border-white/10 bg-slate-950/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-semibold text-white">{label}</div>
        <span className={`rounded-full px-2 py-1 text-[0.65rem] uppercase tracking-[0.24em] ${available ? "bg-emerald-400/15 text-emerald-200" : "bg-slate-700/60 text-slate-400"}`}>
          {available ? "Ready" : "Missing"}
        </span>
      </div>
      <div className="mt-3 text-sm leading-6 text-slate-400">{detail}</div>
      <div className="mt-3 break-all text-xs text-slate-500">{path || "No path available"}</div>
    </div>
  );
}

function LeaderboardTable({ rows, highlightedPipeline }: { rows: EvaluationLeaderboardRow[]; highlightedPipeline: string }) {
  return (
    <div className="overflow-x-auto rounded-[1.5rem] border border-white/10 bg-slate-950/70">
      <table className="min-w-full divide-y divide-white/10 text-sm">
        <thead className="bg-white/5 text-left text-xs uppercase tracking-[0.2em] text-slate-400">
          <tr>
            <th className="px-4 py-3">Rank</th>
            <th className="px-4 py-3">Pipeline</th>
            <th className="px-4 py-3">Weighted</th>
            <th className="px-4 py-3">Raw F1</th>
            <th className="px-4 py-3">Rescaled F1</th>
            <th className="px-4 py-3">Judge %</th>
            <th className="px-4 py-3">Judge pass</th>
            <th className="px-4 py-3">Latency</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {rows.map((row) => {
            const isBest = row.pipeline === highlightedPipeline;
            return (
              <tr key={row.pipeline} className={isBest ? "bg-amber-400/8" : "bg-transparent"}>
                <td className="px-4 py-4 text-slate-200">#{row.rank}</td>
                <td className="px-4 py-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium text-white">{row.pipeline}</span>
                    {isBest ? <span className="rounded-full bg-amber-400/15 px-2 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.24em] text-amber-200">Best</span> : null}
                  </div>
                </td>
                <td className="px-4 py-4 text-slate-200">{(row.hackathon_weighted_score ?? 0).toFixed(2)}</td>
                <td className="px-4 py-4 text-slate-200">{formatNumber(row.avg_bertscore_raw_f1)}</td>
                <td className="px-4 py-4 text-slate-200">{formatNumber(row.avg_bertscore_rescaled_f1)}</td>
                <td className="px-4 py-4 text-slate-200">{formatPercent(row.avg_judge_correctness_pct)}</td>
                <td className="px-4 py-4 text-slate-200">{formatPercent((row.judge_pass_rate ?? 0) * 100)}</td>
                <td className="px-4 py-4 text-slate-200">{formatLatency(row.avg_total_latency_ms)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MetricChart({
  title,
  data,
  dataKey,
  color,
}: {
  title: string;
  data: Array<Record<string, number | string>>;
  dataKey: string;
  color: string;
}) {
  const values = data.map((item) => Number(item[dataKey] ?? 0));
  const maxValue = Math.max(...values, 1);

  return (
    <div className="rounded-[1.5rem] border border-white/10 bg-slate-950/70 p-4 shadow-[0_18px_50px_rgba(0,0,0,0.2)]">
      <h3 className="mb-3 text-sm font-semibold text-white">{title}</h3>
      <div className="h-80">
        <div className="grid h-full grid-cols-3 gap-4">
          {data.map((item, index) => {
            const value = Number(item[dataKey] ?? 0);
            const height = `${Math.max((value / maxValue) * 100, 3)}%`;
            return (
              <div key={`${item.name ?? index}`} className="flex flex-col rounded-[1rem] bg-white/5 p-3 text-center">
                <div className="flex flex-1 items-end justify-center">
                  <div className="w-10 rounded-t-full" style={{ height, backgroundColor: color }} />
                </div>
                <div className="mt-3 text-xs font-medium text-slate-100">{String(item.name ?? `item-${index + 1}`)}</div>
                <div className="mt-1 text-[0.7rem] text-slate-400">{value.toFixed(3)}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ChartCard({ chart }: { chart: EvaluationReportChart }) {
  return (
    <a
      href={apiUrl(chart.url)}
      target="_blank"
      rel="noreferrer"
      className="rounded-[1.25rem] border border-white/10 bg-slate-950/60 p-4 transition hover:border-amber-300/40"
    >
      <div className="text-sm font-semibold text-white">{chart.name}</div>
      <div className="mt-2 text-xs break-all text-slate-500">{chart.path}</div>
    </a>
  );
}

function QuestionComparisonCard({
  questionGroup,
}: {
  questionGroup: { question_id: string; question: string; category: string; difficulty: string; records: BenchmarkRecord[] };
}) {
  return (
    <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/70 p-5">
      <div className="flex flex-col gap-3 border-b border-white/10 pb-4 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="text-[0.65rem] uppercase tracking-[0.28em] text-slate-400">{questionGroup.category}</div>
          <h3 className="mt-2 text-lg font-semibold text-white">{questionGroup.question}</h3>
          <p className="mt-1 text-sm text-slate-400">
            Question ID: {questionGroup.question_id}
            {questionGroup.difficulty ? ` · Difficulty: ${questionGroup.difficulty}` : ""}
          </p>
        </div>
        <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
          {questionGroup.records.length} pipeline runs
        </div>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-3">
        {questionGroup.records.map((record) => (
          <PipelineAnswerCard key={`${record.question_id}-${record.pipeline}`} record={record} />
        ))}
      </div>
    </div>
  );
}

function PipelineAnswerCard({ record }: { record: BenchmarkRecord }) {
  const warnings = buildWarnings(record);
  return (
    <article className="rounded-[1.5rem] border border-white/10 bg-white/5 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[0.65rem] uppercase tracking-[0.28em] text-slate-400">{record.pipeline}</div>
          <div className="mt-1 text-sm text-slate-300">
            {formatLatency(record.total_latency_ms)} · {formatTokens(record.total_tokens)}
          </div>
        </div>
        <div className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-200">{record.retrieval_mode}</div>
      </div>

      <div className="mt-4 rounded-[1.25rem] border border-white/10 bg-slate-950/60 p-4 text-sm leading-7 text-slate-200">
        {record.answer || "No answer returned for this pipeline run."}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 text-xs text-slate-300">
        <MiniMetric label="Raw F1" value={formatNumber(record.bertscore_raw_f1)} />
        <MiniMetric label="Token Δ" value={formatPercent(record.token_reduction_pct_vs_llm_only)} />
        <MiniMetric label="Citations" value={formatPercent(record.citation_correctness * 100)} />
        <MiniMetric label="Mismatch" value={formatPercent(record.answer_context_mismatch * 100)} />
      </div>

      {record.source_titles.length ? (
        <div className="mt-4 rounded-[1.25rem] border border-white/10 bg-slate-950/40 p-3 text-xs leading-6 text-slate-400">
          <div className="mb-1 uppercase tracking-[0.24em] text-slate-500">Sources</div>
          <div>{record.source_titles.slice(0, 4).join(" · ")}</div>
        </div>
      ) : null}

      {warnings.length ? (
        <div className="mt-4 space-y-2">
          {warnings.map((warning) => (
            <div key={warning} className="rounded-[1rem] border border-rose-400/15 bg-rose-400/10 px-3 py-2 text-xs text-rose-100">
              {warning}
            </div>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1rem] border border-white/10 bg-slate-950/40 px-3 py-2">
      <div className="text-[0.6rem] uppercase tracking-[0.2em] text-slate-500">{label}</div>
      <div className="mt-1 text-sm font-medium text-slate-100">{value}</div>
    </div>
  );
}

function buildWarnings(record: BenchmarkRecord) {
  const warnings: string[] = [];
  if (record.error) {
    warnings.push(record.error);
  }
  if (record.fabricated_citation_rate > 0) {
    warnings.push(`Fabricated citation rate: ${formatPercent(record.fabricated_citation_rate * 100)}`);
  }
  if (record.answer_context_mismatch > 0.35) {
    warnings.push(`High answer/context mismatch: ${formatPercent(record.answer_context_mismatch * 100)}`);
  }
  if (!record.retrieval_hit && record.pipeline !== "llm-only") {
    warnings.push("Retrieval pipeline did not hit an expected source.");
  }
  return warnings;
}
